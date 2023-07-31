#!bin/bash

exp_name=$1
metrics_path=$2
stages=(stage1 stage2 stage3)

for stage in ${stages[@]};
    do
        PYTHONWARNINGS="ignore" python -m torch.distributed.launch --nproc_per_node=4 --master_port=4323 \
        ./codeformer/basicsr/train.py --opt codeformer/options/$exp_name  --stage $stage \
        --launcher pytorch --metrics-path $metrics_path
    done
