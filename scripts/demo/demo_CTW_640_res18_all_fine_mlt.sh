#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
                            --exp_name CTW_China_demo \
                            --net resnet18 \
                            --scale 4 \
                            --gpu 0 \
                            --input_size 640 \
                            --num_workers 1 \
                            --cls_threshold 0.05 \
                            --save_dir ./results \
                            --model_type torch \
                            --img_root /home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/0000172.jpg \
                            --model_path /home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/weights/TextBPN_CTW_all_resnet18_best_2.pth \
                            --draw_preds \
                            # --draw_points