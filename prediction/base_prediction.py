from datetime import datetime
import copy
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from cfglib.config import config as cfg
from .prediction_utils import filter_preds, batch_scale_n_shift_dets, nms


##TODO: Make this base normal prediction
class BasePrediction():
    """
    Base Prediction class
    """
    def __init__(self,
                model,
                transform,
                enable_autofocus=False):
        self.model = model
        self.transform = transform
        self.enable_autofocus = enable_autofocus

    def predict_an_image(self, img_name, img):
        '''
        Predict a single image and return prediction results
        '''
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
        ##TODO: priority, Careful padding in transform
        (input_tensor, _), pads = self.transform(copy.deepcopy(chip), return_pads=True)

        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0)

        model_size = self.transform.size
        if isinstance(model_size, int):
                model_size = (model_size - pads[0] - pads[2], model_size - pads[1] - pads[3])
        chip_size = (chip.shape[1], chip.shape[0])
        
        output = self.model(input_tensor)

        # Convert points pred back to original size of chip
        ##TODO: Do we also need for mask map
        ##TODO: Clean the code
        if len(output["py_preds"][-1]) > 0:
            output["py_preds"] = ((output["py_preds"] - np.array(pads[:2])[None, None]) / np.array(model_size)[None, None]) * \
                                    np.array(chip_size)[None, None]
        
        if self.enable_autofocus:
            # Convert focus mask pred back to original size of chip
            focus_mask = output["autofocus_preds"][1]
            focus_mask = (focus_mask >= cfg.focus_threshold).astype(np.uint8)
            focus_mask = cv2.resize(focus_mask, chip_size, interpolation=cv2.INTER_NEAREST)
            output["autofocus_preds"] = focus_mask

        return output

    def _filter_ori_dets(self,
                        chip,
                        preds,
                        to_ori_scale,
                        prev_left_shift,
                        prev_top_shift):
        """
        Filter original detections
        """
        ##TODO: Modify for empty prediction
        py_preds = preds["py_preds"]  # N x num_points x 2
        confidences = preds["confidences"]  # N

        filtered_py_preds, filtered_confidences = filter_preds(py_preds=py_preds,  ##TODO: change name
                                                            confidences=confidences,
                                                            top_k_before_nms=cfg.top_k_before_nms,
                                                            nms_threshold=cfg.nms_threshold,
                                                            top_k_after_nms=cfg.top_k_after_nms)

        ori_py_preds = batch_scale_n_shift_dets(py_preds=filtered_py_preds,
                                                scale=to_ori_scale,
                                                left_shift=prev_left_shift,
                                                top_shift=prev_top_shift)

        return (ori_py_preds, copy.deepcopy(filtered_confidences)), (filtered_py_preds, filtered_confidences)

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
                "chip_with_preds_save_path": _pred["chip_with_preds_save_path"],
                "rank": _pred["rank"],
            }
            flattened_prediction_results.append(pred)
            all_det_preds.append(_pred["ori_dets"])

            sub_chip_preds = _pred["chip_preds"]
            chip_pred_queue.extend(sub_chip_preds)

        all_det_preds = self.stack_dets(all_det_preds)
        return flattened_prediction_results, all_det_preds

    def stack_dets(self, all_det_preds):
        ##TODO: Synamic this (4)
        all_py_preds = [[] for _ in range(4)]
        all_confidences = []

        for (py_preds, confidences) in all_det_preds:
            if len(confidences):
                for i, cur_py_preds in enumerate(py_preds):
                    all_py_preds[i].append(cur_py_preds)
                all_confidences.append(confidences)
        
        if len(all_confidences):
            for i in range(len(all_py_preds)):
                # for p in all_py_preds[i]:
                #     print(p.shape)
                all_py_preds[i] = np.concatenate(all_py_preds[i])
            all_confidences = np.hstack(all_confidences)
        
        return all_py_preds, all_confidences
    
    def draw_predictions(self, image, py_preds):
        if len(py_preds) > 0:
            image_show = image.copy()

            image_show = cv2.polylines(image_show,
                                    [points.astype(int) for points in py_preds], True, self.mask_color, 1)

        return image_show

    def _nms_on_merged_dets(self, all_det_preds):
        """
        Do NMS on the detections of the whole image
        """
        all_py_preds, all_confidences = all_det_preds
        if all_confidences is not None and len(all_confidences) > 0:
            # Do NMS
            if len(all_py_preds[-1]) > 1:
                keep = nms(all_py_preds[-1], all_confidences, cfg.nms_threshold)
                all_py_preds = [all_py_pred[keep] for all_py_pred in all_py_preds]
                all_confidences = all_confidences[keep]
        return all_py_preds, all_confidences