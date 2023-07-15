import os
import argparse
import cv2
import time
import glob
import shutil

from config import Config
from pipeline import Detection


parser = argparse.ArgumentParser("Document Extraction")
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
        else:
            os.mkdir(self.output)

        self.init_modules(config, args.weight)
        self.save_img = True  ##TODO: 

        self.using_rectangle = args.using_rectangle

    def init_modules(self, config, weight):
        self.det_model = Detection(
                config=config,
                weight_path=weight)

    def start(self, img):
        points_list, t = self.det_model(img, return_result=False, using_rectangle=self.using_rectangle)

        results = {
            "points_list": points_list,
            "time": t,
        }
        return results


def get_line_list_result(points_list):
    line_list = []
    
    for points in points_list:
        points = [f"{p:.2f}" for p in points.flatten()]
        points_str = " ".join(points)
        points_str = "0 0.8 " + points_str
        line_list.append(points_str)
    
    return line_list


def main(args, config):
    pipeline = Pipeline(args, config)

    input_list = glob.glob(os.path.join(args.input, "*"))

    start_time = time.time()
    for input_path in input_list:
        input_name = input_path.split(os.sep)[-1].split('.')[0]
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = pipeline.start(img)

        print(f"{input_name}: {result['time']:.3f}")
        line_list = get_line_list_result(result['points_list'])

        with open(os.path.join(args.output, f"{input_name}.txt"), "w") as f:
            f.write("\n".join(line_list))

    end_time = time.time()

    print(f"Executed in {end_time - start_time} s")
    print(f"FPS {len(input_list) / (end_time - start_time):.3f} s")


if __name__ == "__main__":
    config_path = "./config/configs.yaml"
    config = Config(config_path)

    main(args, config)