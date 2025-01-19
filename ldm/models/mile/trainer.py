import os

import pytorch_lightning as pl
import torch
from torchmetrics import JaccardIndex

from ldm.models.mile.constants import BIRDVIEW_COLOURS
from ldm.models.mile.losses import SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss
from ldm.models.mile.models.mile import Mile
from ldm.models.mile.models.preprocess import PreProcess
from ldm.models.mile.models.postprocess import PostProcess
from ldm.util import instantiate_from_config
import scipy
import numpy as np
import cv2
from nuscenes.utils.data_classes import PointCloud, Box
from pyquaternion import Quaternion
from utils.tools import draw_box_in_camera_view,view_points,draw_hdmap_in_camera_view
from PIL import Image
from scipy.spatial import KDTree

class Trainer(pl.LightningModule):
    def __init__(self,model_config,loss_config,ckpt_path=None,ignore_keys=[]):
        super().__init__()
        self.preprocess = PreProcess(loss_config)
        self.postprocess = PostProcess(loss_config)
        self.model = instantiate_from_config(model_config)
        self.action_loss = RegressionLoss(norm=1)
        self.loss_config = loss_config
        self.monitor = 'val_loss'
        if loss_config['MODEL']['TRANSITION']['ENABLED']:
            self.probabilistic_loss = KLLoss(alpha=self.loss_config['LOSSES']['KL_BALANCING_ALPHA'])
        
        if self.loss_config['SEMANTIC_SEG']['ENABLED']:
            self.segmentation_loss = SegmentationLoss(
                use_top_k=self.loss_config['SEMANTIC_SEG']['USE_TOP_K'],
                top_k_ratio=self.loss_config['SEMANTIC_SEG']['TOP_K_RATIO'],
                use_weights=self.loss_config['SEMANTIC_SEG']['USE_WEIGHTS'],
            )

            # self.map_segmentation_loss = SegmentationLoss(
            #     use_top_k=self.loss_config['SEMANTIC_SEG']['USE_TOP_K'],
            #     top_k_ratio=self.loss_config['SEMANTIC_SEG']['TOP_K_RATIO'],
            #     use_weights=self.loss_config['SEMANTIC_SEG']['USE_WEIGHTS'],
            # )

            self.center_loss = SpatialRegressionLoss(norm=2)
            self.offset_loss = SpatialRegressionLoss(norm=1,ignore_index=self.loss_config['INSTANCE_SEG']['IGNORE_INDEX'])

            self.metric_iou_val = JaccardIndex(
                num_classes=self.loss_config['SEMANTIC_SEG']['N_CHANNELS'], reduction='none',
            )
        self.speed_normalization = loss_config['SPEED']['NORMALIZATION']
        if self.loss_config['EVAL']['RGB_SUPERVISION']:
            self.rgb_loss = SpatialRegressionLoss(norm=1)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)

    def init_from_ckpt(self, path,ignore_keys):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,batch,deployment=False):
        batch = self.preprocess(batch)
        output = self.model.forward(batch,deployment=deployment)
        return output
    
    def clear_cache(self,):
        self.model.clear_cache()

    def deployment_forward(self,batch,is_dreaming):
        batch = self.preprocess(batch)
        output = self.model.deployment_forward(batch,is_dreaming)
        output['vel'] = output['vel'] * self.speed_normalization
        bev_labels,box_list,hdmap,route_map = self.postprocess(batch,output)
        # temp = Image.fromarray(hdmap[0].permute(1,2,0).numpy())
        # temp.save(f'all_pics/condition/hdmap.png')
        # temp = Image.fromarray(route_map[0].permute(1,2,0).numpy())
        # temp.save(f'all_pics/condition/route_map.png')
        output['box_list'] = box_list
        output['hdmap'] = hdmap
        output['birdview_label'] = bev_labels
        # colours = torch.tensor(BIRDVIEW_COLOURS,dtype=torch.uint8)
        # target = colours[bev_labels]
        # temp = Image.fromarray(target[0][0].numpy())
        # temp.save(f'all_pics/condition/bev_labels.png')
        output['route_map'] = route_map
        
        return output
    
    def shared_step(self,batch):
        output = self.forward(batch)

        losses = dict()
        action_weight = self.loss_config['LOSSES']['WEIGHT_ACTION']
        losses['vel'] = action_weight * self.action_loss(output['vel'],batch['vel'])
        losses['accel'] = action_weight * self.action_loss(output['accel'],batch['accel'])
        losses['orientation'] = action_weight * self.action_loss(output['orientation'],batch['orientation'])

        if self.loss_config['SEMANTIC_SEG']['ENABLED']:
            for downsampling_factor in [1, 2, 4]:
                bev_segmentation_loss = self.segmentation_loss(
                    prediction=output[f'bev_segmentation_{downsampling_factor}'],
                    target=batch[f'birdview_label_{downsampling_factor}'],
                )
                discount = 1/downsampling_factor
                losses[f'bev_segmentation_{downsampling_factor}'] = discount * self.loss_config['LOSSES']['WEIGHT_SEGMENTATION'] * \
                                                                    bev_segmentation_loss

                center_loss = self.center_loss(
                    prediction=output[f'bev_instance_center_{downsampling_factor}'],
                    target=batch[f'center_label_{downsampling_factor}']
                )
                offset_loss = self.offset_loss(
                    prediction=output[f'bev_instance_offset_{downsampling_factor}'],
                    target=batch[f'offset_label_{downsampling_factor}']
                )

                center_loss = self.loss_config['INSTANCE_SEG']['CENTER_LOSS_WEIGHT'] * center_loss
                offset_loss = self.loss_config['INSTANCE_SEG']['OFFSET_LOSS_WEIGHT'] * offset_loss

                losses[f'bev_center_{downsampling_factor}'] = discount * self.loss_config['LOSSES']['WEIGHT_INSTANCE'] * center_loss
                # Offset are already discounted in the labels
                losses[f'bev_offset_{downsampling_factor}'] = self.loss_config['LOSSES']['WEIGHT_INSTANCE'] * offset_loss

                # map_segmentation_loss = self.map_segmentation_loss(
                #     prediction=output[f'map_segmentation_{downsampling_factor}'],
                #     target=batch[f'map_label_{downsampling_factor}'],
                # )
                # losses[f'map_segmentation_{downsampling_factor}'] = discount * self.loss_config['LOSSES']['WEIGHT_SEGMENTATION'] * map_segmentation_loss

        if self.loss_config['EVAL']['RGB_SUPERVISION']:
            for downsampling_factor in [1, 2, 4]:
                rgb_weight = 0.1
                discount = 1 / downsampling_factor
                rgb_loss = self.rgb_loss(
                    prediction=output[f'rgb_{downsampling_factor}'],
                    target=batch[f'rgb_label_{downsampling_factor}'],
                )
                losses[f'rgb_{downsampling_factor}'] = rgb_weight * discount * rgb_loss
        return losses,output
    
    def training_step(self,batch,batch_idx):
        if batch_idx == self.loss_config['STEPS'] // 2 and self.loss_config['MODEL']['TRANSITION']['ENABLED']:
            self.model.rssm.active_inference = True

        losses,output = self.shared_step(batch)

        self.logging_and_visualisation(batch, output, losses, batch_idx, prefix='train')

        return self.loss_reducing(losses)
    
    def validation_step(self,batch,batch_idx):
        loss, output = self.shared_step(batch)

        if self.loss_config['SEMANTIC_SEG']['ENABLED']:
            seg_prediction = output['bev_segmentation_1'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2)
            self.metric_iou_val(
                seg_prediction.view(-1),
                batch['birdview_label'].view(-1)
            )

        self.logging_and_visualisation(batch, output, loss, batch_idx, prefix='val')
        self.log('val_loss',self.loss_reducing(loss))
        return {'val_loss': self.loss_reducing(loss)}
    
    def logging_and_visualisation(self, batch, output, loss, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', -self.global_step)
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)

        # Visualisation
        # if prefix == 'train':
        #     visualisation_criteria = self.global_step % self.cfg.VAL_CHECK_INTERVAL == 0
        # else:
        #     visualisation_criteria = batch_idx == 0
        # if visualisation_criteria:
        #     self.visualise(batch, output, batch_idx, prefix=prefix)


    def loss_reducing(self,loss):
        total_loss = sum([x for x in loss.values()])
        return total_loss
    
    #TODO: modify
    def validation_epoch_end(self, step_outputs):
        class_names = ['background','human','vehicle','lane','cross','lane_divider']
        if self.loss_config['SEMANTIC_SEG']['ENABLED']:
            scores = self.metric_iou_val.compute()
            for key, value in zip(class_names, scores):
                self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.global_step)
            self.logger.experiment.add_scalar('val_mean_iou', torch.mean(scores), global_step=self.global_step)
            self.metric_iou_val.reset()

    def min_max_normalize(tensor):
        b,n,c,h,w = tensor.shape
        tensor = tensor.view(b*n,c*h*w)
        min_val = tensor.min(dim=1,keepdim=True)[0]
        max_val = tensor.max(dim=1,keepdim=True)[0]
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        normalized_tensor = normalized_tensor.view(b,n,c,h,w)
        return normalized_tensor

    # def get_k_nearest_points(self,points,k=2):
    #     # points (n,2)
    #     kdtree = KDTree(points)
    #     neighbors = []
    #     distances,indices = kdtree.query(points,k=k+1)
    #     for i in range(points.shape[0]):
    #         neighbors.append(indices[i][1:k+1].tolist())
    #     return neighbors

    # def postprocess(self,image,translation,rotation,intrinsics,global2ego_rotation):
    #     b,n,h,w = image.shape
    #     image = image.view(b*n,h,w)
    #     x,y = np.meshgrid(
    #         np.arange(h).astype(np.int16),
    #         np.arange(w).astype(np.int16)
    #     )
    #     instance_image = torch.zeros((b,n,h,w,3)).view(b*n,h,w,3)
    #     camera_image = np.zeros((b*n,900,1600,3)).astype(np.uint8)
    #     colors = {0:torch.tensor((255,0,0)),1:torch.tensor((0,255,0))}
    #     translation = translation.view(b*n,4,1)
    #     rotation = rotation.view(b*n,3,3)
    #     intrinsics = intrinsics.view(b*n,3,3)
    #     global2ego_rotation = global2ego_rotation.view(b*n,4)
    #     camera_intrinsics = np.zeros_like(intrinsics.cpu().numpy())
    #     intrinsics = intrinsics.cpu().numpy()
    #     translation = translation[:,:3].cpu().numpy()
    #     rotation = rotation.cpu().numpy()
    #     global2ego_rotation = global2ego_rotation.cpu().numpy()
    #     camera_intrinsics[:,0] = intrinsics[:,0] #* (1600 / 448)
    #     camera_intrinsics[:,1] = intrinsics[:,1] #* (900 / 256)
    #     camera_intrinsics[:,2] = intrinsics[:,2] * 1

    #     imsize = (1600,900)

    #     flag = False
    #     for i in range(image.shape[0]):
    #         mask = (image[i] == 1)
    #         instance_label,_ = scipy.ndimage.label(mask.cpu().numpy())
    #         # instance_label = torch.from_numpy(instance_label).to(mask.device)
    #         for j in range(1,_+1):
    #             instance_mask = (instance_label == j)
    #             if instance_mask.sum() == 0:
    #                 continue
    #             xc = (x * instance_mask).mean().round()
    #             yc = (y * instance_mask).mean().round()
    #             y_indices,x_indices = np.where(instance_mask)
    #             if len(y_indices) > 0 and len(x_indices) > 0:
    #                 x_min = x_indices.min()
    #                 x_max = x_indices.max()
    #                 y_min = y_indices.min()
    #                 y_max = y_indices.max()
    #                 z_min = -0.15
    #                 z_max = 3
    #                 corners = []
    #                 for z in [z_min,z_max]:
    #                     for x,y in [(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)]:
    #                         corners.append([y,x,z])
    #                 corners = np.array(corners).astype(np.float64).transpose(1,0)
    #                 corners[0,:] = -(corners[0,:] - 192) * 0.4
    #                 corners[1,:] = -(corners[1,:] - 96) * 0.4
    #                 yaw = Quaternion(global2ego_rotation[i]).yaw_pitch_roll[0]
    #                 corners = np.dot(Quaternion(scalar=np.cos(yaw/2),vector=[0,0,np.sin(yaw/2)]).rotation_matrix,corners)
    #                 corners = np.dot(Quaternion(global2ego_rotation[i]).rotation_matrix.T,corners)
    #                 corners = corners - translation[i]
    #                 corners = np.dot(Quaternion._from_matrix(rotation[i]).rotation_matrix.T,corners)

    #                 draw_box_in_camera_view(camera_image[i],corners,camera_intrinsics[i],imsize)
    #                 instance_image[i,y_min:y_max,x_min:x_max] = colors[0]
    #         test_image = Image.fromarray(camera_image[i])
    #         test_image.save("test.png")
    #     instance_image = instance_image.view((b,n,h,w,3))
    #     return instance_image
    
    # def get_map_element(self,image,translation,rotation,intrinsics,global2ego_rotation):
    #     b,n,h,w = image.shape
    #     image = np.array(image.view(b*n,h,w).detach().cpu()).astype(np.uint8)
    #     mask = image == 3
    #     print(mask.sum())
    #     image[mask] = 255
    #     image[~mask] = 0
    #     camera_image = np.zeros((b*n,900,1600,3)).astype(np.uint8)
    #     translation = translation.view(b*n,4,1)
    #     rotation = rotation.view(b*n,3,3)
    #     intrinsics = intrinsics.view(b*n,3,3)
    #     global2ego_rotation = global2ego_rotation.view(b*n,4)
    #     camera_intrinsics = np.zeros_like(intrinsics.cpu().numpy())
    #     intrinsics = intrinsics.cpu().numpy()
    #     translation = translation[:,:3].cpu().numpy()
    #     rotation = rotation.cpu().numpy()
    #     global2ego_rotation = global2ego_rotation.cpu().numpy()
    #     camera_intrinsics[:,0] = intrinsics[:,0] #* (1600 / 448)
    #     camera_intrinsics[:,1] = intrinsics[:,1] #* (900 / 256)
    #     camera_intrinsics[:,2] = intrinsics[:,2] * 1

    #     imsize = (1600,900)
    #     for i in range(image.shape[0]):
    #         positions = np.where(image[i])
    #         positions = np.stack((positions[0],positions[1]),axis=1)
    #         z = np.zeros((positions.shape[0],1))
    #         z[:,:] = -0.15
    #         positions = np.concatenate((positions,z),axis=-1)
    #         neighbors = self.get_k_nearest_points(positions)
    #         positions = positions.transpose(1,0)
    #         positions[0,:] = -(positions[0,:] - 192) * 0.4
    #         positions[1,:] = -(positions[1,:] - 96) * 0.4
    #         yaw = Quaternion(global2ego_rotation[i]).yaw_pitch_roll[0]
    #         positions = np.dot(Quaternion(scalar=np.cos(yaw/2),vector=[0,0,np.sin(yaw/2)]).rotation_matrix,positions)
    #         positions = np.dot(Quaternion(global2ego_rotation[i]).rotation_matrix.T,positions)
    #         positions = positions - translation[i]
    #         positions = np.dot(Quaternion._from_matrix(rotation[i]).rotation_matrix.T,positions)
    #         draw_hdmap_in_camera_view(camera_image[i],positions,camera_intrinsics[i],imsize,neighbors)
    #         test_image = Image.fromarray(camera_image[i])
    #         test_image.save("test.png")
    #     image = np.repeat(image,repeats=3,axis=-1)
    #     image = torch.from_numpy(image).view(b,n,h,w,3)
        
    #     return image
    
        

            

    def infer(self,batch):
        output = self.forward(batch)
        pred = torch.argmax(output['bev_segmentation_1'].detach(),dim=-3)
        infer_out = {}
        colours = torch.tensor(BIRDVIEW_COLOURS,dtype=torch.uint8,device=pred.device)
        target = batch['birdview_label'][:,:,0]
        target = colours[target]
        pred = colours[pred]
        print(target.shape)
        infer_out['birdview_label_pred'] = pred  
        infer_out['birdview_label'] = target
        infer_out['route_map'] = batch['route_map'].permute(0,1,3,4,2)
        print(batch['route_map'].shape)
        # box_list,hdmap = self.postprocess(output)
        # output['box_list'] = box_list
        # output['hdmap'] = hdmap
        # return output
        # infer_out = {}
        # pred = torch.argmax(output['bev_segmentation_1'].detach(),dim=-3)
        # colours = torch.tensor(BIRDVIEW_COLOURS,dtype=torch.uint8,device=pred.device)
        # target = batch['birdview_label'][:,:,0] # b n c h w
        # # pred = colours[pred]
        # # target = colours[target]
        # target = target
        # pred = pred
        # infer_out['birdview_label'] = target
        # translation = batch['ego2cam'][:,:,:,3:]
        # rotation = batch['ego2cam'][:,:,:3,:3]
        # intrinsics = batch['intrinsics']
        # global2ego_rotation = batch['global2ego']
        # infer_out['birdview_label_pred'] = self.postprocess(target,translation,rotation,intrinsics,global2ego_rotation)
        # infer_out['route_map_pred'] = self.get_map_element(pred,translation,rotation,intrinsics,global2ego_rotation)
        # infer_out['birdview_label_pred'] = self.postprocess(pred,translation,rotation,intrinsics)
        # infer_out['image'] = self.preprocess.post_process(batch['image']).permute(0,1,3,4,2)
        # infer_out['image_pred'] = self.preprocess.post_process(output['rgb_1'].detach()).permute(0,1,3,4,2)
        # infer_out['route_map'] = self.preprocess.post_process(batch['route_map']).permute(0,1,3,4,2)
        # infer_out['center_label_1'] = batch['center_label_1'].permute(0,1,3,4,2).repeat(1,1,1,1,3) * 255.
        # infer_out['offset_label_1'] = torch.abs(batch['offset_label_1']).permute(0,1,3,4,2)
        # infer_out['offset_label_pred'] = torch.abs(output['bev_instance_offset_1'].detach()).permute(0,1,3,4,2)
        # infer_out['center_label_pred'] = self.local_search(output['bev_instance_center_1'])
        # infer_out['center_label_pred'] = output['bev_instance_center_1'].detach().permute(0,1,3,4,2).repeat(1,1,1,1,3) * 255.
        # infer_out['center_label_pred'] = self.postprocess(output['bev_instance_center_1'].detach()).permute(0,1,3,4,2)

        # infer_out['route_map_pred'] = self.get_map_element(pred)


        return infer_out


    #TODO: modify
    def visualise(self, batch, output, batch_idx, prefix='train'):
        if not self.loss_config['SEMANTIC_SEG']['ENABLED']:
            return

        target = batch['birdview_label'][:, :, 0]
        pred = torch.argmax(output['bev_segmentation_1'].detach(), dim=-3)

        colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)

        target = colours[target]
        pred = colours[pred]

        # Move channel to third position
        target = target.permute(0, 1, 4, 2, 3)
        pred = pred.permute(0, 1, 4, 2, 3)

        visualisation_video = torch.cat([target, pred], dim=-1).detach()

        # Rotate for visualisation
        visualisation_video = torch.rot90(visualisation_video, k=1, dims=[3, 4])

        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)

    def configure_optimizers(self):
        #  Do not decay batch norm parameters and biases
        # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        def add_weight_decay(model, weight_decay=0.01, skip_list=[]):
            no_decay = []
            decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or any(x in name for x in skip_list):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay},
            ]

        parameters = add_weight_decay(
            self.model,
            self.loss_config['OPTIMIZER']['WEIGHT_DECAY'],
            skip_list=['relative_position_bias_table'],
        )
        weight_decay = 0.
        optimizer = torch.optim.AdamW(parameters, lr=self.loss_config['OPTIMIZER']['LR'], weight_decay=weight_decay)

        # scheduler
        if self.loss_config['SCHEDULER']['NAME'] == 'none':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        elif self.loss_config['SCHEDULER']['NAME'] == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.loss_config['OPTIMIZER']['LR'],
                total_steps=self.loss_config['STEPS'],
                pct_start=self.loss_config['SCHEDULER']['PCT_START'],
            )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]