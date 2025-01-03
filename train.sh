torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_port 5005 \
    main.py --base configs/StreamingSD_cache_6views.yaml --train True 