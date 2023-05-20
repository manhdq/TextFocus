import os
import numpy as np
import cv2
import math

from utils.visualization import visualize, prepare_color
from .base_prediction import BasePrediction
from inference.infer_utils import find_new_size, batch_scale_n_shift


class TextPrediction(BasePrediction):
    def __init__(self, model, foc_chip_gen, priorbox_cfg, data_cfg, demo_cfg, preds_save_dir):
        super().__init__(model, foc_chip_gen, priorbox_cfg, data_cfg, demo_cfg)
        self.draw_preds = demo_cfg['draw_preds']
        self.draw_lm = demo_cfg['draw_lm']
        self.vis_threshold = demo_cfg['vis_threshold']
        self.class_colors, self.lm_color, self.mask_color = prepare_color()
        self.is_review = self.demo_cfg['is_review']
        self.preds_save_dir = preds_save_dir
        
    def recursive_predict(self,
                        img_name,
                        ori_image,
                        chip,
                        rank,
                        base_scale_down,
                        prev_left_shift,
                        prev_top_shift):
        cls_pred, _, loc_pred, lm_pred, focus_mask = self._forward_model(chip)

        to_ori_scale = base_scale_down / \
                    (pow(self.demo_cfg['zoom_in_scale'], max(rank-1, 0)) * \
                    (pow(self.first_round_zoom_in, min(rank, 1))))

        ori_dets, chip_dets = self._filter_ori_dets(chip,
                                                    cls_pred,
                                                    loc_pred,
                                                    lm_pred,
                                                    to_ori_scale,
                                                    prev_left_shift,
                                                    prev_top_shift)

        ori_dets, chip_dets, focus_mask = self._filter_for_first_round(ori_dets, chip_dets, rank, focus_mask,
                                                                    self.demo_cfg['second_round_size_threshold'])
        chip_with_preds_save_path = None

        if self.draw_preds:
            chip_with_preds_save_path = self._draw_pred_on_chip(
                img_name, chip, chip_dets, focus_mask, rank
            )

        chip_preds = []
        ##TODO: Make this recursive and early stop if cant detect any
        if rank < self.demo_cfg['max_focus_rank']:
            chip_height, chip_width = chip.shape[:2]
            chip_preds = self._recursive_pred_on_focus(img_name,
                                                    ori_image,
                                                    focus_mask,
                                                    chip_width,
                                                    chip_height,
                                                    rank,
                                                    base_scale_down,
                                                    to_ori_scale,
                                                    prev_left_shift,
                                                    prev_top_shift)
        
        result = {
            "chip_with_preds_save_path": chip_with_preds_save_path,
            "rank": rank,
            "chip_preds": chip_preds,
            "ori_dets": ori_dets,
        }
        return result

    def _filter_for_first_round(self, ori_dets, chip_dets, rank, focus_mask, size_threhold):
        if rank > 0:
            sqrt_sizes = [math.sqrt((det[2] - det[0]) * (det[3] - det[1])) for det in ori_dets]
            valid_size = [size < size_threhold for size in sqrt_sizes]
            ori_dets = ori_dets[valid_size]
            chip_dets = chip_dets[valid_size]
        else:
            focus_mask = np.ones_like(focus_mask)
        return ori_dets, chip_dets, focus_mask

    def _save_ori_img_pred(self, img_name, ori_image, detections):
        """
        Save predictions on original image
        """
        ori_pred_save_path = None
        if detections is not None:
            # Filter detections to visualize with vis_threshold
            vis_detections = detections[detections[:, 4] >= self.vis_threshold]
            if self.draw_preds:
                ##TODO: Do we need `is_review` variable
                ori_pred_save_path = self._draw_on_ori_image(img_name=img_name,
                                                            ori_image=ori_image,
                                                            bbox=vis_detections[:, :4],
                                                            conf=vis_detections[:, 4],
                                                            label=vis_detections[:, -1],
                                                            landm=vis_detections[:, 5:15],
                                                            valid_range=self.demo_cfg["draw_valid_range"],
                                                            suffix="preds",
                                                            is_review=self.is_review)

        return ori_pred_save_path

    def _draw_pred_on_chip(self, img_name, chip, dets, focus_mask, rank):
        """
        Draw and save predictions to chips
        """
        chip_with_preds_save_path = None

        # Filter detections to visualize with vis_threshold
        vis_dets = dets[dets[:, 4] >= self.vis_threshold]
        # Show lm
        if not self.draw_lm:
            vis_dets[:, 5:15] = -1
        vis_bbox, vis_conf, vis_lm = vis_dets[:, :4], vis_dets[:, 4], vis_dets[:, 5:15]

        # Draw preds on img
        for i, color_key in enumerate(self.class_colors.keys()):
            chip = visualize(image=chip,
                            bbox=vis_bbox[vis_dets[:, -1] == i + 1],
                            bbox_color=self.class_colors[color_key],
                            conf=vis_conf[vis_dets[:, -1] == i + 1],
                            lm=vis_lm[vis_dets[:, -1] == i + 1],
                            lm_color=self.lm_color,
                            mask=focus_mask,
                            mask_color=self.mask_color)
        chip_with_preds_save_path = os.path.abspath(
            os.path.join(self.preds_save_dir, img_name.split('.')[0], f'{img_name}_with_preds_{rank}_{self.pred_idx}.jpg')
        )
        cv2.imwrite(chip_with_preds_save_path,
                    chip, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        self.pred_idx += 1  # Increase pred_idx for the next saved prediction
        return chip_with_preds_save_path

    def _draw_on_ori_image(self,
                        img_name,
                        ori_image,
                        bbox,
                        conf,
                        label,
                        landm,
                        valid_range,
                        suffix,
                        is_review=False):
        """
        Draw and save bbox, landmarks and mask to original image
        """
        ori_image_height, ori_image_width = ori_image.shape[:2]
        # Resize original image and ground truth before drawing
        new_size, new_scale_down = find_new_size(image_width=ori_image_width,
                                                image_height=ori_image_height,
                                                valid_range=valid_range,
                                                base_scale_down=1)
        resized_ori_image = cv2.resize(
            ori_image, new_size, interpolation=self.demo_cfg['interpolation'])
        
        resized_bbox = batch_scale_n_shift(batch_coord=bbox,
                                           scale=1 / new_scale_down,
                                           left_shift=0,
                                           top_shift=0)
        resized_landm = batch_scale_n_shift(batch_coord=landm,
                                            scale=1 / new_scale_down,
                                            left_shift=0,
                                            top_shift=0)
        
        ##TODO: Make this dynamic later
        resized_landm[:, :] = -1
        # Draw bboxes and landmarks on original image
        for i, color_key in enumerate(self.class_colors.keys()):
            vis_conf = conf[label == i + 1] if conf is not None else conf
            resized_ori_image = visualize(image=resized_ori_image,
                                          bbox=resized_bbox[label == i + 1],
                                          bbox_color=self.class_colors[color_key],
                                          conf=vis_conf,
                                          lm=resized_landm[label == i + 1],
                                          lm_color=self.lm_color,
                                          mask=None,
                                          mask_color=self.mask_color,
                                          is_review=is_review)

        save_path = os.path.abspath(
            os.path.join(self.preds_save_dir, img_name.split('.')[0], f'{img_name}_ori_{suffix}.jpg'))
        cv2.imwrite(save_path, resized_ori_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return save_path

    def _post_process_dets(self, img_name, ori_image, all_det_preds):
        '''
        After got detections, post-process the detections
        by doing NMS, counting predictions, making decision and drawing image
        '''
        ##TODO: Delete this `text_count` variable
        merged_dets = self._nms_on_merged_dets(all_det_preds)
        count_results = [0]
        if merged_dets is not None:
            count_results = [len(merged_dets)]
        ori_pred_save_path = self._save_ori_img_pred(
            img_name, ori_image, merged_dets
        )
        return count_results, ori_pred_save_path