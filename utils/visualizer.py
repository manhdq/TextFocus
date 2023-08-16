import os
import os.path as osp
import cv2
import numpy as np


class Visualizer:
    def __init__(self, vis_path):
        self.vis_path = vis_path
        if not osp.exists(vis_path):
            os.makedirs(vis_path)

    def process(self, img_metas, outputs):
        img_path = img_metas['img_path'][0]
        img_name = img_metas['img_name'][0]
        bboxes = outputs['bboxes']
        if 'words' in outputs:
            words = outputs['words']
        else:
            words = [None] * len(bboxes)

        img = cv2.imread(img_path)
        for bbox, word in zip(bboxes, words):
            cv2.drawContours(img, [bbox.reshape(-1, 2)], -1, (0, 255, 0), 2)
            if word is not None:
                pos = np.min(bbox.reshape(-1, 2), axis=0)
                cv2.putText(img, word, (pos[0], pos[1]),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imwrite(osp.join(self.vis_path, '%s.jpg' % img_name), img)


def visualize(image, points_group, points_color, draw_points, boundary_color, 
            mask, mask_color, confidences):
    '''
    Visualize bbox, landmarks and focus mask
    '''
    new_image = image.copy()
    image_height, image_width = new_image.shape[:2]

    # Draw boundary
    boundary_size = max(min(image_height, image_width) // 1000, 1) + 1
    for points in points_group:
        new_image = cv2.polylines(new_image,
                        [points.astype(int)], True, boundary_color, boundary_size)  # Draw last level

    ##TODO: Make this optional
    # Draw confidences
    # Get center
    centers = [points.mean(axis=1).astype(int) for points in points_group]
    text_face = cv2.FONT_HERSHEY_DUPLEX
    text_scale = 0.4
    text_thickness = 1
    for confidence, center in zip(confidences, centers):
        text = f"{confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, text_face, text_scale, text_thickness)
        text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))
        cv2.putText(new_image, text, text_origin, text_face, text_scale, (0,0,0), text_thickness, cv2.LINE_AA)
    
    if mask is not None:
        # Draw focus mask
        focus_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        overlay = new_image.copy()
        # Blue overlay for interested object (255, 0, 0)
        overlay[:, :, 0][focus_mask > 0.2] = mask_color[0] # Blue
        overlay[:, :, 1][focus_mask > 0.2] = mask_color[1] # Green
        overlay[:, :, 2][focus_mask > 0.2] = mask_color[2] # Red
        new_image = cv2.addWeighted(overlay, 0.2, new_image, 0.8, 0)
    
    return new_image


def visualize_detect(image, points_group, boundary_color, confidences):
    '''
    Visualize bbox, landmarks and focus mask
    '''
    new_image = image.copy()
    image_height, image_width = new_image.shape[:2]

    # Draw boundary
    boundary_size = max(min(image_height, image_width) // 1000, 1) + 1
    for points in points_group:
        new_image = cv2.polylines(new_image,
                        [points.astype(int)], True, boundary_color, boundary_size)  # Draw last level

    ##TODO: Make this optional
    # Draw confidences
    # Get center
    centers = [points.mean(axis=1).astype(int) for points in points_group]
    text_face = cv2.FONT_HERSHEY_DUPLEX
    text_scale = 0.4
    text_thickness = 1
    for confidence, center in zip(confidences, centers):
        text = f"{confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, text_face, text_scale, text_thickness)
        text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))
        cv2.putText(new_image, text, text_origin, text_face, text_scale, (0,0,0), text_thickness, cv2.LINE_AA)
    
    return new_image


def visualize_focus_mask(image, mask, mask_color):
    '''
    Visualize bbox, landmarks and focus mask
    '''
    new_image = image.copy()
    image_height, image_width = new_image.shape[:2]
    
    if mask is not None:
        # Draw focus mask
        focus_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        overlay = new_image.copy()
        # Blue overlay for interested object (255, 0, 0)
        overlay[:, :, 0][focus_mask > 0.2] = mask_color[0] # Blue
        overlay[:, :, 1][focus_mask > 0.2] = mask_color[1] # Green
        overlay[:, :, 2][focus_mask > 0.2] = mask_color[2] # Red
        new_image = cv2.addWeighted(overlay, 0.2, new_image, 0.8, 0)
    
    return new_image