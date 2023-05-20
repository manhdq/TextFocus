'''
Basic Prediction class
'''
from datetime import datetime

import cv2
import numpy as np

from utils import PriorBox
from utils.nms import nms
from utils.misc import filter_preds
from inference.grid_generator import GridGenerator
from inference.infer_utils import batch_scale_n_shift, batch_scale_n_shift_dets, RejectException
from .prediction_utils import log_info, aggregate_prediction_result


class BasePrediction():
    '''
    Basic Prediction class
    '''
    def __init__(self,
                 model,
                 foc_chip_gen,
                 priorbox_cfg,
                 data_cfg,
                 demo_cfg):
        self.model = model
        self.foc_chip_gen = foc_chip_gen
        self.priorbox_cfg = priorbox_cfg
        self.data_cfg = data_cfg
        self.demo_cfg = demo_cfg
        self.first_round_zoom_in = self.demo_cfg['first_round_zoom_in']
        self.grid_gen = GridGenerator(max_valid_size=self.demo_cfg['max_valid_size'],
                                    grid_threshold=self.demo_cfg['grid_threshold'],
                                    overlap_ratio=self.demo_cfg['overlap_ratio'],
                                    base_scale_down=self.demo_cfg['scale_down'],
                                    valid_range=self.demo_cfg['valid_range'],
                                    interpolation=self.demo_cfg['interpolation'],
                                    max_chip_size=self.demo_cfg['max_chip_size'])

    def predict_an_image(self,
                       img_name,
                       img):
        '''
        Predict a single image and return prediction results
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
                    'max_focus_rank': self.demo_cfg['max_focus_rank'],
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
            
            count_results, ori_pred_save_path = \
                self._post_process_dets(img_name, img, all_det_preds)
            log_info(start_pred_time, img_name, rank=None)
            reformated_prediction_result = aggregate_prediction_result(img_name=img_name,
                                                            flattened_infer_results=flattened_prediction_results,
                                                            count_results=count_results,
                                                            start_pred_time=start_pred_time,
                                                            ori_pred_save_path=ori_pred_save_path,
                                                            max_focus_rank=self.demo_cfg['max_focus_rank'],
                                                            ori_image_shape=img.shape)
            return reformated_prediction_result

        except RejectException:
            log_info(start_pred_time, img_name, rank=None)
            return aggregate_prediction_result(img_name=img_name,
                                          flattened_infer_results=[],
                                          count_results=None,
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
        raise NotImplementedError

    def _post_process_dets(self, item_id, ori_image, groundtruth, all_det_preds):
        '''
        After got detections, post-process the detections
        by doing NMS, counting predictions, making decision and drawing image
        '''
        raise NotImplementedError

    def _forward_model(self, chip):
        """
        Forward a chip through model
        and post-process predictions
        """
        input_tensor = (chip[None, ...] - self.data_cfg['rgb_mean']).transpose(0, 3, 1, 2)
        loc_pred, cls_pred, conf_pred, lm_pred, foc_pred = self.model(input_tensor)

        focus_mask = foc_pred[1]
        focus_mask = (focus_mask >= self.demo_cfg['focus_threshold']).astype(np.uint8)
        return cls_pred, conf_pred, loc_pred, lm_pred, focus_mask

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
        Cut focus chip and do predicction on this chip
        """
        ###TODO: Must refactor this function
        chip_preds = []
        
        # Crop sub chips from the input chip by using the focus mask
        chip_coords = self.foc_chip_gen(focus_mask, chip_width, chip_height)
        for chip_coord in chip_coords:
            for tile_coord in self.grid_gen.gen_from_chip(chip_coord, rank):
                # Convert chip coordinates to original coordinates by
                # scaling chip coordinates
                # and shift chip coordinates to match the top-left of the original image
                resized_bbox = batch_scale_n_shift(batch_coord=np.expand_dims(tile_coord, axis=0),
                                                scale=to_ori_scale,
                                                left_shift=prev_left_shift,
                                                top_shift=prev_top_shift)

                ori_x1, ori_y1, ori_x2, ori_y2 = resized_bbox[0]

                # Crop interested region on original image
                ori_chip_crop = ori_image[round(ori_y1):round(ori_y2),
                                        round(ori_x1):round(ori_x2), :]

                zoom_scale = self.demo_cfg['zoom_in_scale'] if rank > 0 else self.first_round_zoom_in
                zoom_in_x1, zoom_in_y1, zoom_in_x2, zoom_in_y2 = \
                    list(map(lambda x: x * zoom_scale, tile_coord))
                zoom_in_chip_w = round(zoom_in_x2 - zoom_in_x1)
                zoom_in_chip_h = round(zoom_in_y2 - zoom_in_y1)

                zoom_in_chip_crop = cv2.resize(ori_chip_crop,
                                            (zoom_in_chip_w, zoom_in_chip_h),
                                            interpolation=self.demo_cfg['interpolation'])
                
                chip_pred = self.recursive_predict(img_name,
                                                ori_image,
                                                zoom_in_chip_crop,
                                                rank + 1,
                                                base_scale_down,
                                                ori_x1,
                                                ori_y1)
                chip_preds.append(chip_pred)
                
        return chip_preds

    def _filter_ori_dets(self,
                         chip,
                         cls_pred,
                         loc_pred,
                         lm_pred,
                         to_ori_scale,
                         prev_left_shift,
                         prev_top_shift):
        '''
        Filter original detections
        '''
        # TODO: Must refactor this function
        box_scale, lm_scale, priors = self._create_priors(chip)
        dets = filter_preds(cls_pred=cls_pred,
                            loc_pred=loc_pred,
                            lm_pred=lm_pred,
                            box_scale=box_scale,
                            lm_scale=lm_scale,
                            priors=priors,
                            variance=self.demo_cfg['variance'],
                            top_k_before_nms=self.demo_cfg['top_k_before_nms'],
                            nms_threshold=self.demo_cfg['nms_threshold'],
                            top_k_after_nms=self.demo_cfg['top_k_after_nms'],
                            nms_per_class=self.demo_cfg['nms_per_class'])
        
        ori_dets = batch_scale_n_shift_dets(dets=dets,
                                            scale=to_ori_scale,
                                            left_shift=prev_left_shift,
                                            top_shift=prev_top_shift)
        return ori_dets, dets

    def _flatten_prediction_result(self, prediction_results):
        """
        Flatten the recursive predictions
        """
        flattened_prediction_results = []
        chip_pred_queue = [*prediction_results]
        all_det_preds = []

        while len(chip_pred_queue) > 0:
            _pred = chip_pred_queue.pop(0)

            pred = {
                'chip_with_preds_save_path': _pred['chip_with_preds_save_path'],
                'rank': _pred['rank']
            }
            flattened_prediction_results.append(pred)
            all_det_preds.append(_pred['ori_dets'])

            sub_chip_preds = _pred['chip_preds']
            chip_pred_queue.extend(sub_chip_preds)

        return flattened_prediction_results, np.vstack(all_det_preds)

    def _create_priors(self, chip):
        """
        Create priors boxes
        """
        chip_height, chip_width, _ = chip.shape
        box_scale = np.array([chip_width, chip_height] * 2)
        lm_scale = np.array([chip_width, chip_height] * 5)

        # Create prior_boxes
        priorbox = PriorBox(
            cfg=self.priorbox_cfg, image_size=(chip_height, chip_width), to_tensor=False
        )
        chip_priors = priorbox.generate()

        return box_scale, lm_scale, chip_priors

    def _nms_on_merged_dets(self, all_det_preds):
        """
        Do NMS on the detections of the whole image
        """
        if all_det_preds is not None and len(all_det_preds) > 0:
            # Do NMS
            if self.demo_cfg['nms_per_class']:
                max_classes = all_det_preds[:, -1]
                all_classes = np.unique(max_classes)
                all_dets = np.empty((0, 16)).astype(all_det_preds.dtype)
                for class_id in all_classes:
                    class_inds = max_classes == class_id
                    class_det_preds = all_det_preds[class_inds]
                    if len(class_det_preds) > 1:
                        keep = nms(class_det_preds[:, :5], self.demo_cfg['nms_threshold'])
                        class_det_preds = class_det_preds[keep]
                    all_preds = np.concatenate((all_dets, class_det_preds), axis=0)
                all_det_preds = all_preds
            else:
                if len(all_det_preds) > 1:
                    keep = nms(all_det_preds[:, :5], self.demo_cfg['nms_threshold'])
                    all_det_preds = all_det_preds[keep]
            return all_det_preds
        else:
            return None