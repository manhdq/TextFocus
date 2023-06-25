import os
import cv2
import numpy as np
import math
from datetime import datetime
from shapely.geometry import Polygon

from cfglib.config import config as cfg
from utils.visualize import visualize
from .base_prediction import BasePrediction
from .grid_generator import GridGenerator
from .prediction_utils import RejectException, batch_scale_n_shift_dets, find_new_size, \
                            log_info, aggregate_prediction_result


class TextBPNFocusPrediction(BasePrediction):
    def __init__(self, model, transform, foc_chip_gen):
        super().__init__(model, transform, enable_autofocus=True)
        ##TODO: eliminate redundancy
        self.first_row_zoom_in = cfg.first_row_zoom_in
        self.foc_chip_gen = foc_chip_gen
        self.grid_gen = GridGenerator(max_valid_size=cfg.max_valid_size,
                                    grid_threshold=cfg.grid_threshold,
                                    overlap_ratio=cfg.overlap_ratio,
                                    base_scale_down=cfg.scale_down,
                                    valid_range=cfg.valid_range,
                                    interpolation=cfg.interpolation,
                                    max_chip_size=cfg.max_chip_size)


    def predict_an_image(self, img_name, img):
        '''
        Predict a single image with autofocus option and return prediction results
        '''
        start_pred_time = datetime.now()
        ori_img_h, ori_img_w = img.shape[:2]
        self.pred_idx = 0

        scaled_down_tiles, base_scale_down = self.grid_gen.gen_from_ori_img(img)
        if scaled_down_tiles is None:
            print(f'Ignore "{img_name}" due to the image size ({ori_img_w}x{ori_img_h})!')
            return {
                img_name: {
                    'ori_image_with_merged_preds_save_path': None,
                    'max_focus_rank': cfg.max_focus_rank,
                    'num_text': 0,
                    'predictions': [],
                    'ori_image_shape': img.shape,
                    'prediction_time': 0
                }
            }

        reformated_prediction_result = dict()
        try:
            prediction_results = []
            for tile in scaled_down_tiles:
                prediction_result = self.recursive_predict(img_name,
                                                img,
                                                tile['image'],
                                                rank=0,
                                                base_scale_down=base_scale_down,
                                                prev_left_shift=tile['prev_left_shift'],
                                                prev_top_shift=tile['prev_top_shift'])
                prediction_results.append(prediction_result)

            flattened_prediction_results, all_det_preds = self._flatten_prediction_result(prediction_results)

            ori_pred_save_path = \
                    self._post_process_dets(img_name, img, all_det_preds)
            log_info(start_pred_time, img_name, rank=None)
            reformated_prediction_result = aggregate_prediction_result(img_name=img_name,
                                                                    flattened_prediction_results=flattened_prediction_results,
                                                                    start_pred_time=start_pred_time,
                                                                    ori_pred_save_path=ori_pred_save_path,
                                                                    max_focus_rank=cfg.max_focus_rank,
                                                                    ori_image_shape=img.shape)
            return reformated_prediction_result

        except RejectException:
            log_info(start_pred_time, img_name, rank=None)
            return aggregate_prediction_result(img_name=img_name,
                                          flattened_prediction_results=[],
                                          start_pred_time=start_pred_time,
                                          ori_pred_save_path=None,
                                          max_focus_rank=self.demo_cfg['max_focus_rank'],
                                          ori_image_shape=img.shape)
        
    def recursive_predict(self,
                        img_name,
                        ori_image,
                        chip,
                        rank,
                        base_scale_down,
                        prev_left_shift,
                        prev_top_shift):
        outputs = self._forward_model(chip)

        to_ori_scale = base_scale_down / \
                    (pow(cfg.zoom_in_scale, max(rank - 1, 0)) * \
                    (pow(self.first_row_zoom_in, min(rank, 1))))

        ori_dets, chip_dets = self._filter_ori_dets(chip,
                                                    outputs,
                                                    to_ori_scale,
                                                    prev_left_shift,
                                                    prev_top_shift)

        focus_mask = outputs["autofocus_preds"]
        ori_dets, chip_dets, focus_mask = self._filter_for_first_round(ori_dets, chip_dets, rank, focus_mask,
                                                                    cfg.second_round_size_threshold)
        chip_with_preds_save_path = None

        ##TODO: priority
        # if self.draw_preds:
        #     chip_with_preds_save_path = self._draw_pred_on_chip(
        #         img_name, chip, chip_dets, focus_mask, rank, levels=[3]
        #     )
        
        chip_preds = []
        ##TODO: Make this recursive and early stop if cant detect any
        if rank < cfg.max_focus_rank:
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

    def _recursive_pred_on_focus(self, 
                                img_name, 
                                ori_image, 
                                focus_mask, 
                                chip_width, 
                                chip_height,
                                rank,
                                base_scale_down,
                                to_ori_scale,
                                prev_left_shift,
                                prev_top_shift):
        """
        Cut focus chip and do prediction on this chip
        """
        ##TODO: Must redactor this function
        chip_preds = []

        # Crop sub chips from the input chip by using the focus mask
        chip_coords = self.foc_chip_gen(focus_mask, chip_width, chip_height)
        for chip_coord in chip_coords:
            for tile_coord in self.grid_gen.gen_from_chip(chip_coord, rank):
                # Convert chip coordinates to original coordinates by
                # scaling chip coordinates
                # and shift chip coordinates to match the top-left of the original image
                resized_bbox = batch_scale_n_shift_dets(py_preds=[np.expand_dims(tile_coord, axis=0).astype(float)],
                                                scale=to_ori_scale,
                                                left_shift=prev_left_shift,
                                                top_shift=prev_top_shift)[0]
                
                ori_x1, ori_y1, ori_x2, ori_y2 = resized_bbox[0]

                # Crop interested region on original image
                ori_chip_crop = ori_image[round(ori_y1):round(ori_y2),
                                        round(ori_x1):round(ori_x2), :]

                zoom_scale = cfg.zoom_in_scale if rank > 0 else self.first_row_zoom_in
                zoom_in_x1, zoom_in_y1, zoom_in_x2, zoom_in_y2 = \
                    list(map(lambda x: x * zoom_scale, tile_coord))
                zoom_in_chip_w = round(zoom_in_x2 - zoom_in_x1)
                zoom_in_chip_h = round(zoom_in_y2 - zoom_in_y1)

                zoom_in_chip_crop = cv2.resize(ori_chip_crop,
                                            (zoom_in_chip_w, zoom_in_chip_h),
                                            interpolation=cfg.interpolation)

                chip_pred = self.recursive_predict(img_name,
                                                ori_image,
                                                zoom_in_chip_crop,
                                                rank + 1,
                                                base_scale_down,
                                                ori_x1,
                                                ori_y1)
                chip_preds.append(chip_pred)

        return chip_preds

    def _filter_for_first_round(self, ori_dets, chip_dets, rank, focus_mask, size_threshold):
        # rank = 1
        ##TODO: Current code assume there are one sample per demo or inference
        ##TODO: Clean code
        if rank > 0:
            ori_py_preds, ori_confidences = ori_dets
            chip_py_preds, chip_confidences = chip_dets
            polygons = [Polygon(points) for points in ori_py_preds[-1]]  # get points for last level
            sqrt_sizes = [math.sqrt(polygon.area)for polygon in polygons]
            valid_size = [size < size_threshold for size in sqrt_sizes]

            ori_py_preds = [ori_cur_py_preds[valid_size] for ori_cur_py_preds in ori_py_preds]
            ori_confidences = ori_confidences[valid_size]
            chip_py_preds = [chip_cur_py_preds[valid_size] for chip_cur_py_preds in chip_py_preds]
            chip_confidences = chip_confidences[valid_size]
            
            ori_dets = (ori_py_preds, ori_confidences)
            chip_dets = (chip_py_preds, chip_confidences)
        else:
            focus_mask = np.ones_like(focus_mask)
        
        return ori_dets, chip_dets, focus_mask

    def _draw_pred_on_chip(self, img_name, chip, dets, focus_mask, rank, levels=[0, 1, 2, 3]):
        """
        Draw and save predictions to chips
        """
        chip_with_preds_save_path = None

        py_preds, confidences = dets

        # Filter detections for visualize with vis_threshold
        selected_sample = confidences >= self.vis_threshold
        vis_py_preds = [cur_py_preds[selected_sample] for cur_py_preds in py_preds]
        vis_confidences = confidences[selected_sample]

        # Draw preds on img
        chip = visualize(image=chip,
                        points_group=vis_py_preds,
                        points_color=self.lm_color,
                        draw_points=self.draw_points,
                        boundary_color=self.boundary_color,
                        mask=focus_mask,
                        mask_color=self.mask_color,
                        confidences=vis_confidences,
                        levels=levels)
        chip_with_preds_save_path = os.path.abspath(
            os.path.join(self.preds_save_dir, img_name.split('.')[0], f"{img_name.split('.')[0]}_with_preds_{rank}_{self.pred_idx}.jpg")
        )
        os.makedirs(os.path.join(self.preds_save_dir, img_name.split('.')[0]), exist_ok=True)
        cv2.imwrite(chip_with_preds_save_path,
                    chip[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        self.pred_idx += 1  # Increase pred_idx for the next saved prediction  ##TODO: For just 1 image
        return chip_with_preds_save_path

    def _post_process_dets(self, img_name, ori_image, all_det_preds):
        '''
        After got detections, post-process the detections
        by doing NMS, counting predictions, making decision and drawing image
        '''
        merged_dets = self._nms_on_merged_dets(all_det_preds)
        ori_pred_save_path = self._save_ori_img_pred(
            img_name, ori_image, merged_dets
        )
        return ori_pred_save_path

    def _save_ori_img_pred(self, img_name, ori_image, detections):
        """
        Save predictions on original image
        """
        ori_pred_save_path = None
        all_py_preds, all_confidences = detections
        if all_confidences is not None and len(all_confidences):
            # Filter detections to visualize with vis_threshold
            valid_conf = all_confidences >= self.vis_threshold
            all_py_preds = [cur_all_py_preds[valid_conf] for cur_all_py_preds in all_py_preds]
            all_confidences = all_confidences[valid_conf]

        if self.draw_preds:
            ori_pred_save_path = self._draw_on_ori_image(img_name=img_name,
                                                        ori_image=ori_image,
                                                        all_py_preds=all_py_preds,
                                                        all_confidences=all_confidences,
                                                        valid_range=cfg.draw_valid_range,
                                                        suffix="preds")
        ##TODO: priority
        self.save_txt_result(img_name, all_py_preds, all_confidences)
        return ori_pred_save_path

    def _draw_on_ori_image(self,
                        img_name,
                        ori_image,
                        all_py_preds,
                        all_confidences,
                        valid_range,
                        suffix,):
        """
        Draw and save landmarks and mask to original image
        """
        ori_image_height, ori_image_width = ori_image.shape[:2]
        # Resize original image and ground truth before drawing
        new_size, new_scale_down = find_new_size(image_width=ori_image_width,
                                                image_height=ori_image_height,
                                                valid_range=valid_range,
                                                base_scale_down=1)
        resized_ori_image = cv2.resize(
            ori_image, new_size, interpolation=cfg.interpolation)
        
        if len(all_confidences):
            resized_py_preds = batch_scale_n_shift_dets(all_py_preds,
                                                        scale=1 / new_scale_down,
                                                        left_shift=0,
                                                        top_shift=0)
            # Draw preds on original image
            resized_ori_image = visualize(image=resized_ori_image,
                                        points_group=resized_py_preds,
                                        points_color=self.lm_color,
                                        draw_points=self.draw_points,
                                        boundary_color=self.boundary_color,
                                        mask=None,
                                        mask_color=self.mask_color,
                                        confidences=all_confidences,
                                        levels=[3])  ##TODO:

        save_path = os.path.abspath(
            os.path.join(self.preds_save_dir, img_name.split('.')[0], f"{img_name.split('.')[0]}_ori_{suffix}.jpg")
        )
        os.makedirs(os.path.join(self.preds_save_dir, img_name.split('.')[0]), exist_ok=True)
        cv2.imwrite(save_path,
                    resized_ori_image[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return save_path
