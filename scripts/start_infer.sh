#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python -m infer \
    --cfg-path ./configs/retinafocus_configs_infer.yml \
    --model-path /home/tungnguyendinh/review_assistant/project/snapshots/large_mr_no_mr_data_max_0.4_gen_min_size/best_loss.pth \
    --image-dir /home/tungnguyendinh/review_assistant/project/demo_upload/ \
    --save-dir infer_results/ra_errors \
    --model-type torch \
    --test-json-path /home/tungnguyendinh/review_assistant/support/datasets/ndt_pixta/ra_errors/pixta_test_ori.json \
    # --image-dir /home/tungnguyendinh/review_assistant/support/datasets/ndt_pixta/ra_errors/test_ori \
    # --early-stop
    # --model-type torch \
