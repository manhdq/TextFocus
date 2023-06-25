from datetime import datetime
import copy
import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from cfglib.config import config as cfg
from utils.visualize import visualize
from .prediction_utils import filter_preds, batch_scale_n_shift_dets, nms, log_info


def prepare_color():
    '''
    Prepare color to visualize
    '''
    lm_color = (0, 142, 248)              # lm point - blue
    boundary_color = (0, 246, 249)        # boundary color - cyan
    mask_color = (255, 0, 255)            # mask color - purple
    return lm_color, boundary_color, mask_color


##TODO: Make this base normal prediction
class BasePrediction():
    """
    Base Prediction class
    """
    def __init__(self,
                model,
                transform,
                enable_autofocus=False):
        self.draw_preds = cfg.draw_preds
        self.draw_points = cfg.draw_points
        self.vis_threshold = cfg.vis_threshold
        self.lm_color, self.boundary_color, self.mask_color = prepare_color()
        self.preds_save_dir = cfg.save_dir

        self.model = model
        self.transform = transform
        self.enable_autofocus = enable_autofocus

    def predict_an_image(self, img_name, img):
        '''
        Predict a single image and return prediction results
        '''
        start_pred_time = datetime.now()
        img_h, img_w = img.shape[:2]

        outputs = self._forward_model(img)
        py_preds = outputs["py_preds"]
        confidences = outputs["confidences"]
        
        if self.draw_preds:
            pred_save_path = self._draw_on_ori_image(img_name=img_name,
                                                    ori_image=img,
                                                    all_py_preds=py_preds,
                                                    all_confidences=confidences,
                                                    valid_range=cfg.draw_valid_range,
                                                    suffix="preds")

        self.save_txt_result(img_name, py_preds, confidences)
        log_info(start_pred_time, img_name, rank=None)
        return {
            img_name: {
                "save_path": pred_save_path,
                "predictions": [(py_preds, confidences)],
                "ori_image_shape": img.shape,
                "prediction_time": (datetime.now() - start_pred_time).total_seconds()
            }
        }
    
    def save_txt_result(self, img_name, all_py_preds, all_confidences):
        last_py_preds = all_py_preds[-1]
        lines = []
        save_txt_path = os.path.abspath(
            os.path.join(self.preds_save_dir, "txt_preds", f"{img_name.split('.')[0]}.txt")
        )
        for py_pred, confidence in zip(last_py_preds, all_confidences):
            py_pred_text = " ".join([f"{point[0]:.2f} {point[1]:.2f}" for point in py_pred])
            line = f"0 {confidence:.2f} {py_pred_text}"
            lines.append(line)
        
        with open(save_txt_path, "w") as f:
            f.write("\n".join(lines))

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
        save_image = ori_image.copy()
        if len(all_confidences):
            save_image = visualize(image=save_image,
                                points_group=all_py_preds,
                                points_color=self.lm_color,
                                draw_points=self.draw_points,
                                boundary_color=self.boundary_color,
                                mask=None,
                                mask_color=self.mask_color,
                                confidences=all_confidences,
                                levels=[3])  ##TODO:
        save_path = os.path.abspath(
            os.path.join(self.preds_save_dir, f"{img_name.split('.')[0]}.jpg")
        )
        cv2.imwrite(save_path,
                    save_image[..., ::-1])
        return save_path

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