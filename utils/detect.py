from pse import decode as pse_decode
from config_lib.config import config as cfg


class TextDetector(object):

    def __init__(self, model):
        # evaluation mode
        self.model = model
        model.eval()
        # parameter
        self.scale = cfg.scale
        self.threshold = cfg.threshold

    def detect(self, image, img_show):
        # get model output
        preds = self.model.forward(image)
        preds, boxes, contours = pse_decode(preds[0], self.scale, self.threshold)

        output = {
            'image': image,
            'tr': preds,
            'bbox': boxes
        }
        return contours, output

