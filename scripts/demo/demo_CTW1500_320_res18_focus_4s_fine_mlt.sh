#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
                            --exp_name CTW1500_demo \
                            --net resnet18 \
                            --enable_autofocus \
                            --autofocus_dont_care_low 3 \
                            --autofocus_dont_care_high 150 \
                            --autofocus_small_threshold 50 \
                            --scale 4 \
                            --gpu 0 \
                            --input_size 320 \
                            --num_workers 1 \
                            --cls_threshold 0.7 \
                            --save_dir ./results \
                            --model_type torch \
                            --img_root /home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/Images/test \
                            --model_path /home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/weights/TextBPN_resnet18_best.pth \
                            --draw_preds \
                            # --draw_points