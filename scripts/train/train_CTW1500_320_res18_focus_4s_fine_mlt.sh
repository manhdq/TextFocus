#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
                            --exp_name CTW1500 \
                            --net resnet18 \
                            --enable_autofocus \
                            --autofocus_dont_care_low 3 \
                            --autofocus_dont_care_high 150 \
                            --autofocus_small_threshold 50 \
                            --scale 4 \
                            --max_epoch 660 \
                            --batch_size 1 \
                            --gpu 0 \
                            --input_size 320 \
                            --optim Adam --lr 0.0001 \
                            --num_workers 1 \
                            --load_memory False