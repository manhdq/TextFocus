
CUDA_VISIBLE_DEVICES=0 \
python demo.py \
    --img-root /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test_ori/img396.jpg \
    --model-path /home/ubuntu/Documents/working/pixtaVN/RA/AutoFocus_TT/ckpts/best_det_map.pth \
    --cfg-path configs/retinafocus_totaltext_configs_demo.yml \
    --save-dir tmp \
    --model-type torch