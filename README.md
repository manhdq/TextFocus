1. Pull data to folder data follow by README.md in dataset folder

2. Create conda anv

    Required: torch (latest) with cuda toolkit 11.8, mmcv==1.3.1

```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```shell
    pip install -U openmim
    mim install mmcv==1.3.1
```
3. Install libraries

```shell
    conda env create -n text 
    pip install -r requirement.txt
```

4. Train
If use pretrained, please uncoment 'config/pan/pan_r18_ctw_finetune.py#L55' else:

```shell
    sh train.sh
```
5. Test

```shell
    sh test.sh
```


