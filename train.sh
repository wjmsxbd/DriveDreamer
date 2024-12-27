torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    main.py --base configs/slow_fast_learning.yaml --train True 