#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python -m infer \
    --cfg-path ./configs/retinafocus_totaltext_configs_infer.yml \
    --model-path /home/ubuntu/Documents/working/pixtaVN/RA/AutoFocus_TT/ckpts/best_det_map.pth \
    --image-dir /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test_ori \
    --save-dir infer_results/totaltext_errors \
    --model-type torch \
    --test-json-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test_ori.json \
    # --image-dir /home/tungnguyendinh/review_assistant/support/datasets/ndt_pixta/ra_errors/test_ori \
    # --early-stop
    # --model-type torch \
