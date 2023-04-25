
python data/preprocess/focus_generator.py \
    --input-path final_with_chips_22_39_filtered.json \
    --output-path final_with_mask_22_39_filtered.json \
    --dont-care-low 5 \
    --dont-care-high 90 \
    --small-threshold 25 \
    --label-w 32 \
    --label-h 32 \
    --stride 16 \
