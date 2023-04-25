
python data/preprocess/chip_generator.py \
    --input-train-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/COCO/train.json \
    --input-test-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/COCO/test.json \
    --root-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/yolo/images \
    --ori-img-test-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test_ori/ \
    --ori-json-test-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test_ori.json \
    --out-train-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/train.json \
    --out-test-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/test.json \
    --chip-save-path /home/ubuntu/Documents/working/pixtaVN/RA/data/total-text/focus_data/ \
    --valid-range 0 \
    --c-stride 32 \
    --mapping-threshold 0.45 \
    --training-size 448 \
    --n-threads 2 \
    --use-neg 0 \
