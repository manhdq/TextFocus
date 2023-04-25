import torch
from torch import optim
import torch.utils.data as data
import yaml

from retinafocus import RetinaFocus
from data import FocusRetinaDataset, focus_retina_collate
from data.augmentations import DetectionAugmentation
from data.preprocess import FocusGenerator
from loss import RetinaFocusLoss
from utils import Trainer, PriorBox
from utils.parser import train_parser
from utils.misc import show_params_gflops, set_random_seed

set_random_seed(42, use_cuda=True)


def main():
    ##### GET CONFIGURATION #####
    args = train_parser()

    with open(args.cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        train_cfg = cfg['train']
        data_cfg = cfg['data']
        model_cfg = cfg['model']
        retinafocus_cfg = model_cfg['retinafocus']
        priorbox_cfg = model_cfg['priorbox']

        data_cfg['num_classes'] = retinafocus_cfg['retinaface']['num_classes']

    ##### PREPARING DATASETS #####
    print('\nPreparing datasets...')
    data_aug = DetectionAugmentation(training_size=data_cfg['image_size'],
                                     brighten_param=data_cfg['brighten_param'],
                                     contrast_param=data_cfg['contrast_param'],
                                     saturate_param=data_cfg['saturate_param'],
                                     hue_param=data_cfg['hue_param'],
                                     resize_methods=data_cfg['resize_methods'],
                                     rgb_mean=data_cfg['rgb_mean'],
                                     pre_scales=data_cfg['pre_scales'],
                                     use_albumentations=data_cfg['use_albumentations'])
    focus_gen = FocusGenerator(dont_care_low=data_cfg['dont_care_low'],
                               dont_care_high=data_cfg['dont_care_high'],
                               small_threshold=data_cfg['small_threshold'],
                               stride=retinafocus_cfg['autofocus']['stride'])
    train_dataset = FocusRetinaDataset(json_path=args.train_json_path,
                                       root_path=args.image_train_dir,
                                       aug=data_aug,
                                       focus_gen=focus_gen,
                                       phase='train',
                                       data_cfg=data_cfg,
                                       mixup_cfg=data_cfg['mixup'],
                                       train_bbox_iof_threshold=data_cfg['train_bbox_iof_threshold'],
                                       train_min_num_landmarks=data_cfg['train_min_num_landmarks'])
    train_dataset[0]
    exit()
    val_dataset = FocusRetinaDataset(json_path=args.val_json_path,
                                     root_path=args.image_val_dir,
                                     aug=data_aug,
                                     focus_gen=focus_gen,
                                     phase='val',
                                     data_cfg=data_cfg)

    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=data_cfg['batch_size'],
                                       shuffle=True,
                                       num_workers=data_cfg['num_workers'],
                                       pin_memory=False,
                                       collate_fn=focus_retina_collate)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=data_cfg['val_batch_size'],
                                     shuffle=False,
                                     num_workers=data_cfg['val_num_workers'],
                                     pin_memory=False,
                                     collate_fn=focus_retina_collate)

    ##### CREATE MODEL #####
    model = RetinaFocus(cfg=retinafocus_cfg,
                        retinaface_weights_path=args.retinaface_weights_path,
                        exclude_top_retinaface=args.exclude_top_retinaface)
    ##TODO: Modify for dynamic device selection
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    show_params_gflops(model,
                       (3, data_cfg['image_size'], data_cfg['image_size']),
                       print_layer=False)

    ##### CREATE PRIOR BOXES #####
    priorbox = PriorBox(cfg=priorbox_cfg,
                        image_size=(data_cfg['image_size'],
                                    data_cfg['image_size']),
                        to_tensor=True)
    priors = priorbox.generate()
    priors = priors.cuda()

    ##### CREATE CRITERION #####
    criterion = RetinaFocusLoss(num_classes=retinafocus_cfg['retinaface']['num_classes'],
                                neg_pos=train_cfg['negpos_ratio'],
                                variance=train_cfg['variance'],
                                cfg=train_cfg)

    ##### CREATE OPTIMIZER #####
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=train_cfg['lr'],
                        momentum=0.9,
                        weight_decay=5e-4)

    ##### CREATE TRAINER AND START THE TRAINING PROCESS #####
    print(f'Training model using pre_scales', data_cfg['pre_scales'])
    trainer = Trainer(
        model=model,
        priors=priors,
        criterion=criterion,
        optimizer=optimizer,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        checkpoint_path=args.checkpoint_path
    )
    trainer.fit(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()