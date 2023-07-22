model = dict(
    type='PAN',
    backbone=dict(
        type='resnet18',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v1',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='PA_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v1',
            feature_dim=4,
            loss_weight=0.25
        )
    ),
    focus_head=dict(
        type='Focus_Head',
        in_channels=64,                       # 64, 128, 256, 512
        focus_layer_choice=0,                 # 0: up1;  1: up2;  2: up3;  3: up4
        loss_focus=dict(
            type="FocalFocusLoss",
            loss_weight=0.5,
            ignore_index=-1
        )
    )
)
data = dict(
    batch_size=2,
    root_dir = '/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/',
    train_data_dir = 'Images/chip_for_train/',
    test_data_dir = 'Images/chip_for_test/',
    train_gt_dir = 'gt/chip_for_train/',
    test_gt_dir = 'gt/chip_for_test/',
    train=dict(
        type='PAN_CTW',
        split='train',
        is_transform=True,
        img_size=320,
        short_size=320,
        kernel_scale=0.7,
        read_type='cv2'
    ),
    test=dict(
        type='PAN_CTW',
        split='test',
        short_size=320,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=6,
    optimizer='Adam',
    pretrain='pretrained/pan_r18_synth.pth.tar'
)
test_cfg = dict(
    min_score=0.88,
    min_area=16,
    bbox_type='poly',
    result_path='outputs/submit_ctw/'
)

using_autofocus = True
autofocus = dict(
    dont_care_low=3,
    dont_care_high=200,
    small_threshold=50,
    stride=4,
)

using_tensorboard=True
log_dir="logs"