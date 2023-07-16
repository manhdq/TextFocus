import os
import argparse
import cv2
import time
import shutil

from config import Config
from pipeline import Detection


parser = argparse.ArgumentParser("Document Extraction")
parser.add_argument("--config_path")
parser.add_argument("--input", help="Path to single image to be scanned")
parser.add_argument("--output", default="./results", help="Path to output folder")
parser.add_argument("--weight", type=str)
parser.add_argument("--using_rectangle", action="store_true")
args = parser.parse_args()


class Pipeline:
    def __init__(self, args, config):
        self.output = args.output
        if os.path.exists(self.output):
            shutil.rmtree(self.output)
            os.mkdir(self.output)

        self.init_modules(config, args.weight)
        self.save_img = True  ##TODO: 

        self.using_rectangle = args.using_rectangle

    def init_modules(self, config, weight):
        self.det_model = Detection(
                config=config,
                weight_path=weight)

    def start(self, img):
        boxes, t, img = self.det_model(img, return_result=True, using_rectangle=self.using_rectangle)

        saved_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.output, "test.jpg"), saved_img)

        results = {
            "bboxes": boxes,
            "time": t,
        }
        return results


if __name__ == "__main__":
    config_path = args.config_path
    config = Config(config_path)
    pipeline = Pipeline(args, config)
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    pipeline.start(img)
    end_time = time.time()

    print(f"Executed in {end_time - start_time} s")