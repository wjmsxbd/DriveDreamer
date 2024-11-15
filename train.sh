torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=mayuexin01:5001 \
    --node_rank=0\
    main.py --base configs/StreamingSD_cache.yaml --train True 