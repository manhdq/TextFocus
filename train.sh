# Compile Cython
cd ./models/post_processing/pa/
python3 setup.py build_ext --inplace
cd ../boxgen/
python3 setup.py build_ext --inplace
cd ../../../
echo Done!

CUDA_VISIBLE_DEVICES=0 python3 train.py config/pan/pan_r18_ctwchina_finetune.py
    
