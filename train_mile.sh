torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    main.py --base configs/MILE.yaml --train True