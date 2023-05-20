'''
Functions to visualize image
'''
import cv2


def prepare_color():
    '''
    Prepare color to visualize
    '''
    class_colors = {
        'text': (0, 255, 0),     # Text - green
    }
    lm_color = (0, 255, 0)              # lm point - green
    mask_color = (255, 255, 0)          # mask color - cyan
    return class_colors, lm_color, mask_color


def visualize(image, bbox, bbox_color, conf, lm, lm_color, mask, mask_color, is_review=False):
    '''
    Visualize bbox, landmarks and focus mask
    '''
    new_image = image.copy()
    image_height, image_width = new_image.shape[:2]

    if bbox is not None and lm is not None:
        for i, (box, points) in enumerate(zip(bbox, lm)):
            # Draw bbox
            b_x1, b_y1, b_x2, b_y2 = map(round, box[0:4])
            cv2.rectangle(new_image, (b_x1, b_y1), (b_x2, b_y2), bbox_color, 2)

            if not is_review:
                # Draw confidence
                if conf is not None:
                    text_conf = "{:.2f}".format(conf[i])
                    cv2.putText(new_image,
                                text_conf,
                                (b_x1, b_y1 - 4),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.5,
                                bbox_color)

                # Draw landms
                if points[0] != -1:
                    for j in range(0, 10, 2):
                        cv2.circle(new_image, (int(points[j]), int(points[j + 1])), 1, lm_color, 2)


    if mask is not None:
        # Draw focus mask
        focus_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        overlay = new_image.copy()
        # Blue overlay for interested object (255, 0, 0)
        overlay[:, :, 0][focus_mask == 1] = mask_color[0] # Blue
        overlay[:, :, 1][focus_mask == 1] = mask_color[1] # Green
        overlay[:, :, 2][focus_mask == 1] = mask_color[2] # Red
        new_image = cv2.addWeighted(overlay, 0.1, new_image, 0.9, 0)

    return new_image
