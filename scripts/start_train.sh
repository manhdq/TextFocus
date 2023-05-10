#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python -m train \
    --train-json-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/train.json \
    --val-json-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test.json \
    --image-train-dir /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/train \
    --image-val-dir /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test \
    --cfg-path ./configs/retinafocus_totaltext_configs_train.yml \
    # --checkpoint-path /home/tungnguyendinh/review_assistant/project/snapshots/mr_large_mr_data_max_0.4_no_human/best_loss.pth
    # --retinaface-weights-path weights/Resnet50_Final.pth \
    # --exclude-top-retinaface
