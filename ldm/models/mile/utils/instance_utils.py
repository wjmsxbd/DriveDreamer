import torch

#TODO: check ignore_index
def convert_instance_mask_to_center_and_offset_label(instance_label,ignore_index=255,sigma=3):
    batch_size,seq_len,h,w = instance_label.shape
    center_label = torch.zeros(batch_size,seq_len,1,h,w,device=instance_label.device)
    offset_label = ignore_index * torch.ones(batch_size,seq_len,2,h,w,device=instance_label.device)
    x,y = torch.meshgrid(
        torch.arange(h,dtype=torch.float,device=instance_label.device),
        torch.arange(w,dtype=torch.float,device=instance_label.device)
    )

    for b in range(batch_size):
        num_instances = int(instance_label[b].max())
        for instance_id in range(1,num_instances+1):
            for t in range(seq_len):
                instance_mask = (instance_label[b,t] == instance_id)
                if instance_mask.sum()==0:
                    continue

                xc = x[instance_mask].mean().round().long()
                yc = y[instance_mask].mean().round().long()

                off_x = xc - x
                off_y = yc - y
                g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
                center_label[b,t,0] = torch.maximum(center_label[b,t,0],g)
                offset_label[b,t,0,instance_mask] = off_x[instance_mask]
                offset_label[b,t,1,instance_mask] = off_y[instance_mask]
    return center_label,offset_label
