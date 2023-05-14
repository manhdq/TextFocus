import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.special import softmax

from utils.misc import filter_batch_preds, remove_prefix
from utils.cycle_lr import OneCycle
from metrics import FocusRetinaMetricsCalculator


def moving_avg(avg, update, alpha):
    return (alpha * avg) + ((1 - alpha) * update)


class TensorBoardWriter(object):
    def __init__(self, log_dir: str, purge_step: int = 0):
        self.log_dir = log_dir
        self.purge_step = purge_step

    def __enter__(self):
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=self.purge_step
        )

        return self.writer

    def __exit__(self, type, value, traceback):
        self.writer.close()


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 priors: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_cfg: dict,
                 data_cfg: dict,
                 model_cfg: dict,
                 device: str,
                 checkpoint_path: None):
        self.model = model
        self.priors = priors
        self.priors_np = priors.detach().cpu().numpy()
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.device = device

        self.start_from_epoch = self.train_cfg['start_from_epoch']
        self.stop_at_epoch = \
            self.train_cfg['start_from_epoch'] + \
            self.train_cfg['epochs']
        self.log_dir = os.path.join('./runs', self.train_cfg['id'])
        self.save_dir = os.path.join('./snapshots', self.train_cfg['id'])

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Initialize monitoring params
        self.best_val_loss = np.inf
        self.best_det_map = 0
        self.best_foc_dice = 0
        self.best_foc_iou = 0
        self.best_foc_diff = np.inf
        ##TODO: Move alpha to config file ??
        self.alpha = 0.9  # Mean over 10 iters

        if self.train_cfg['use_rop_scheduler']:
            self.rop_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=self.train_cfg['rop_factor'],
                patience=self.train_cfg['rop_patience'],
                min_lr=1e-7,
                verbose=True
            )
        elif self.train_cfg['use_steplr_scheduler']:
            self.steplr_scheduler = optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=self.train_cfg['sl_step_size'],
                gamma=self.train_cfg['sl_gamma']
            )

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.metric_cal = FocusRetinaMetricsCalculator(iou_threshold=self.train_cfg['iou_threshold'],
                                                       mask_conf_threshold=self.train_cfg['mask_conf_threshold'],
                                                       stride=self.model_cfg['retinafocus']['autofocus']['stride'])

    def fit(self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader):
        with TensorBoardWriter(self.log_dir, purge_step=self.start_from_epoch) as writer:
            ##TODO: fix _log_graph() func for multi-gpu
            if self.train_cfg['log_graph'] and torch.cuda.device_count() <= 1:
                self._log_graph(train_dataloader, writer)

            print('\nTraining model...')
            iteration = len(train_dataloader) * self.train_cfg['epochs']
            ##TODO: Can replace this lr scheduler
            onecycle = OneCycle(num_iteration=iteration,
                                max_lr=self.train_cfg['lr'],
                                start_div=self.train_cfg['1cycle_start_div'],
                                end_div=self.train_cfg['1cycle_end_div'])
            for epoch in range(self.start_from_epoch, self.stop_at_epoch):
                self._fit_an_epoch(
                    train_dataloader, val_dataloader, writer, epoch, onecycle)

    def _log_graph(self, dataloader, writer):
        image_batch, _, _, _, _ = next(iter(dataloader))
        image_batch = image_batch.to(self.device)
        writer.add_graph(self.model, (image_batch))

    def _load_checkpoint(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path)

        print('\nLoading checkpoint...')
        states = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
        print('|`-- Loading model...')
        print('+--------------------')
        states['model'] = remove_prefix(states['model'], 'module.')
        self.model.load_state_dict(states['model'])
        if not self.train_cfg['load_weights_only']:
            print('|`-- Loading optimizer...')
            self.optimizer.load_state_dict(states['optimizer'])
            print('|`-- Loading best val loss...')
            self.best_val_loss = states['best_val_loss']
            print('|`-- Loading best detection mAP...')
            self.best_det_map = states['best_det_map']
            print('|`-- Loading best focus DICE...')
            self.best_foc_dice = states['best_foc_dice']
            print('|`-- Loading best focus IOU...')
            self.best_foc_iou = states['best_foc_iou']
            print(' `-- Loading best focus diff...')
            self.best_foc_diff = states['best_foc_diff']

    def _save_checkpoint(self,
                         new_val_loss,
                         new_det_map,
                         new_foc_dice,
                         new_foc_iou,
                         new_foc_diff,
                         epoch):
        found_better_val_loss = new_val_loss < self.best_val_loss
        found_better_det_map = new_det_map > self.best_det_map
        found_better_foc_dice = new_foc_dice > self.best_foc_dice
        found_better_foc_iou = new_foc_iou > self.best_foc_iou
        found_better_foc_diff = new_foc_diff < self.best_foc_diff

        self.best_val_loss = np.minimum(self.best_val_loss, new_val_loss)
        self.best_det_map = np.maximum(self.best_det_map, new_det_map)
        self.best_foc_dice = np.maximum(self.best_foc_dice, new_foc_dice)
        self.best_foc_iou = np.maximum(self.best_foc_iou, new_foc_iou)
        self.best_foc_diff = np.minimum(self.best_foc_diff, new_foc_diff)

        states = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_det_map': self.best_det_map,
            'best_foc_dice': self.best_foc_dice,
            'best_foc_iou': self.best_foc_iou,
            'best_foc_diff': self.best_foc_diff,
            'train_cfg': self.train_cfg,
            'data_cfg': self.data_cfg,
            'model_cfg': self.model_cfg
        }

        if self.train_cfg['save_best_loss'] and found_better_val_loss:
            print('  |__ Found a better checkpoint based on val loss -> Saving...')
            torch.save(states,
                       os.path.join(self.save_dir, 'best_loss.pth'))

        if self.train_cfg['save_best_det_map'] and found_better_det_map:
            print('  |__ Found a better checkpoint based on detection mAP -> Saving...')
            torch.save(states,
                       os.path.join(self.save_dir, 'best_det_map.pth'))

        if self.train_cfg['save_best_foc_dice'] and found_better_foc_dice:
            print('  |__ Found a better checkpoint based on focus DICE -> Saving...')
            torch.save(states,
                       os.path.join(self.save_dir, 'best_foc_dice.pth'))

        if self.train_cfg['save_best_foc_iou'] and found_better_foc_iou:
            print('  |__ Found a better checkpoint based on focus IOU -> Saving...')
            torch.save(states,
                       os.path.join(self.save_dir, 'best_foc_iou.pth'))

        if self.train_cfg['save_best_foc_diff'] and found_better_foc_diff:
            print('  |__ Found a better checkpoint based on focus diff -> Saving...')
            torch.save(states,
                       os.path.join(self.save_dir, 'best_foc_diff.pth'))

        if self.train_cfg['save_latest']:
            torch.save(states,
                       os.path.join(self.save_dir, 'latest.pth'))

        if self.train_cfg['save_all_epochs']:
            torch.save(states,
                       os.path.join(self.save_dir, 'epoch_{0}.pth'.format(epoch+1)))

    def _convert_to_numpy(self, item):
        return item.detach().cpu().numpy()

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def update_mom(self, mom):
        for g in self.optimizer.param_groups:
            g['momentum'] = mom

    def _fit_an_epoch(self, train_dataloader, val_dataloader, writer, epoch, onecycle):
        # Training process
        self.model.train()

        # Initialize a dictionary to store numeric values
        running = {
            'loss': {
                'loc': 0,
                'cls': 0,
                'conf': 0,
                'lm': 0,
                'foc': 0,
                'sum': 0
            },
        }

        train_pbar = tqdm(train_dataloader)
        train_pbar.desc = "* Epoch {0}".format(epoch + 1)
        train_iters = len(train_dataloader)

        for batch_idx, (image_batch, targets_batch, mask_batch, _, _) in enumerate(train_pbar):
            image_batch = image_batch.to(self.device)
            targets_batch = [target_batch.to(self.device)
                            for target_batch in targets_batch]
            mask_batch = mask_batch.to(self.device)

            lr, mom = onecycle.calc()
            self.update_lr(lr)
            # self.update_lr(mom) # Update mom => loss increase to infinity

            preds = self.model(image_batch)

            loss = dict()
            loss['loc'], loss['cls'], loss['conf'], loss['lm'], loss['foc'] = \
                self.criterion(preds[:-1],
                            self.priors,
                            targets_batch,
                            preds[-1],
                            mask_batch)

            loss['sum'] = self.train_cfg['loc_weight'] * loss['loc'] \
                + self.train_cfg['cls_weight'] * loss['cls'] \
                + self.train_cfg['conf_weight'] * loss['conf'] \
                + self.train_cfg['lm_weight'] * loss['lm'] \
                + self.train_cfg['foc_weight'] * loss['foc']

            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            # Compute running losses
            for key in running['loss'].keys():
                running['loss'][key] = moving_avg(
                    running['loss'][key], loss[key].item(), self.alpha)

            train_pbar.set_postfix({
                'loss_loc': running['loss']['loc'],
                'loss_cls': running['loss']['cls'],
                'loss_conf': running['loss']['conf'],
                'loss_lm': running['loss']['lm'],
                'loss_foc': running['loss']['foc'],
                'loss_sum': running['loss']['sum']
            })

            # Log to TensorBoard
            global_iter_idx = epoch * train_iters + batch_idx
            if (global_iter_idx % self.train_cfg['log_every']) == 0 \
                    or batch_idx == (train_iters - 1):

                for key in running.keys():
                    for subkey in running[key]:
                        writer.add_scalar('{0}/train_{1}'.format(key, subkey),
                                          running[key][subkey],
                                          global_iter_idx)

        states = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_det_map': self.best_det_map,
            'best_foc_dice': self.best_foc_dice,
            'best_foc_iou': self.best_foc_iou,
            'best_foc_diff': self.best_foc_diff,
            'train_cfg': self.train_cfg,
            'data_cfg': self.data_cfg,
            'model_cfg': self.model_cfg
        }
        torch.save(states, os.path.join(self.save_dir, 'latest.pth'))

        # Validation process
        self.model.eval()

        with torch.no_grad():
            # Initialize a dictionary to store 1d arrays
            batch_val = {
                'loss': {
                    'loc': np.zeros(len(val_dataloader)),
                    'cls': np.zeros(len(val_dataloader)),
                    'conf': np.zeros(len(val_dataloader)),
                    'lm': np.zeros(len(val_dataloader)),
                    'foc': np.zeros(len(val_dataloader)),
                    'sum': np.zeros(len(val_dataloader))
                }
            }

            num_items = 0

            val_pbar = tqdm(val_dataloader)
            val_pbar.desc = '\__ Validating'

            det_predictions = []
            foc_predictions = []
            targets = []
            masks = []

            for batch_val_idx, (image_batch, targets_batch, mask_batch, _, _) in enumerate(val_pbar):
                image_batch = image_batch.to(self.device)
                targets_batch = [target_batch.to(self.device)
                                for target_batch in targets_batch]
                mask_batch = mask_batch.to(self.device)

                image_height, image_width = image_batch.shape[-2:]
                box_scale = np.array([image_width, image_height] * 2)
                lm_scale = np.array([image_width, image_height] * 5)

                preds = self.model(image_batch)

                loc_preds, cls_preds, conf_preds, lm_preds, foc_preds = \
                    map(self._convert_to_numpy, preds)

                # Softmax the predictions
                ##TODO: fix this double softmax
                cls_preds = softmax(cls_preds, axis=-1)
                conf_preds = softmax(conf_preds, axis=-1)
                foc_preds = softmax(foc_preds, axis=1)

                batch_dets = filter_batch_preds(cls_preds=cls_preds,
                                                loc_preds=loc_preds,
                                                lm_preds=lm_preds,
                                                box_scale=box_scale,
                                                lm_scale=lm_scale,
                                                priors=self.priors_np,
                                                variance=self.train_cfg['variance'],
                                                top_k_before_nms=self.train_cfg['top_k_before_nms'],
                                                nms_threshold=self.train_cfg['nms_threshold'],
                                                top_k_after_nms=self.train_cfg['top_k_after_nms'],
                                                nms_per_class=self.train_cfg['nms_per_class'])

                foc_predictions.extend(foc_preds)
                det_predictions.extend(batch_dets)
                targets.extend([target_batch.detach().cpu().numpy()
                                for target_batch in targets_batch])
                masks.extend(mask_batch.detach().cpu().numpy())

                val_loss = dict()
                val_loss['loc'], val_loss['cls'], val_loss['conf'], val_loss['lm'], val_loss['foc'] = \
                    self.criterion(preds[:-1],
                                   self.priors,
                                   targets_batch,
                                   preds[-1],
                                   mask_batch)
                val_loss['sum'] = self.train_cfg['loc_weight'] * val_loss['loc'] \
                    + self.train_cfg['cls_weight'] * val_loss['cls'] \
                    + self.train_cfg['conf_weight'] * val_loss['conf'] \
                    + self.train_cfg['lm_weight'] * val_loss['lm'] \
                    + self.train_cfg['foc_weight'] * val_loss['foc']

                local_batch_size = image_batch.size(0)
                num_items += local_batch_size

                # Compute sum batch losses
                for key in batch_val['loss'].keys():
                    batch_val['loss'][key][batch_val_idx] = val_loss[key].item(
                    ) * local_batch_size

                val_pbar.set_postfix({
                    'loss_loc': batch_val['loss']['loc'][batch_val_idx] / local_batch_size,
                    'loss_cls': batch_val['loss']['cls'][batch_val_idx] / local_batch_size,
                    'loss_conf': batch_val['loss']['conf'][batch_val_idx] / local_batch_size,
                    'loss_lm': batch_val['loss']['lm'][batch_val_idx] / local_batch_size,
                    'loss_foc': batch_val['loss']['foc'][batch_val_idx] / local_batch_size,
                    'loss_sum': batch_val['loss']['sum'][batch_val_idx] / local_batch_size
                })

            all_predictions = list(zip(det_predictions, foc_predictions))
            all_targets = list(zip(targets, masks))
            print(' \__ Computing metrics...')
            metrics = self.metric_cal(all_predictions,
                                      all_targets,
                                      (self.data_cfg['image_size'], self.data_cfg['image_size']))

            mean_val = dict()
            for key in batch_val.keys():
                mean_val[key] = dict()
                for subkey in batch_val[key].keys():
                    mean_val[key][subkey] = np.sum(
                        batch_val[key][subkey]) / num_items

            if self.train_cfg['use_rop_scheduler']:
                self.rop_scheduler.step(mean_val['loss']['sum'])
            elif self.train_cfg['use_steplr_scheduler']:
                self.steplr_scheduler.step()

            # Log to TensorBoard
            for key in mean_val.keys():
                for subkey in mean_val[key].keys():
                    writer.add_scalar('{0}/val_{1}'.format(key, subkey),
                                      mean_val[key][subkey], epoch)
            for key in metrics.keys():
                for subkey in metrics[key].keys():
                    if isinstance(metrics[key][subkey], dict):
                        for subsubkey in metrics[key][subkey].keys():
                            writer.add_scalar('{0}/val_{1}_{2}'.format(key, subkey, subsubkey),
                                              metrics[key][subkey][subsubkey], epoch)
                    else:
                        writer.add_scalar('{0}/val_{1}'.format(key, subkey),
                                          metrics[key][subkey], epoch)

            # Save checkpoint
            self._save_checkpoint(
                new_val_loss=mean_val['loss']['sum'],
                new_det_map=metrics['detection']['map'],
                new_foc_dice=metrics['focus']['dice'],
                new_foc_iou=metrics['focus']['iou'],
                new_foc_diff=metrics['focus']['diff'],
                epoch=epoch
            )