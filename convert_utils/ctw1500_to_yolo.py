import os
import glob
import shutil
import argparse
import numpy as np
from lxml import etree as ET
from PIL import Image

from data_check import check_ann


def get_box(pts, img_size):
    w, h = img_size

    x1 = pts[:, 0].min()
    y1 = pts[:, 1].min()
    x2 = pts[:, 0].max()
    y2 = pts[:, 1].max()

    xn = (x1 + x2) / 2 / w
    yn = (y1 + y2) / 2 / h
    wn = (x2 - x1) / w
    hn = (y2 - y1) / h
    return (xn, yn, wn, hn)


def get_text_index(line_infos):
    for i, line in enumerate(line_infos):
        if line[:4] == "####":
            return i
    raise


def main(args):
    img_train_root = os.path.join(args.data_root, "Images", "train_images")
    img_test_root = os.path.join(args.data_root, "Images", "test_images")
    ann_train_root = os.path.join(args.data_root, "gt", "train_labels")
    ann_test_root = os.path.join(args.data_root, "gt", "test_labels")

    img_train_dest = os.path.join(args.save_dir, "Images", "train")
    img_test_dest = os.path.join(args.save_dir, "Images", "test")
    ann_train_dest = os.path.join(args.save_dir, "gt", "train")
    ann_test_dest = os.path.join(args.save_dir, "gt", "test")

    os.makedirs(img_train_dest, exist_ok=True)
    os.makedirs(img_test_dest, exist_ok=True)
    os.makedirs(ann_train_dest, exist_ok=True)
    os.makedirs(ann_test_dest, exist_ok=True)

    for ann_train_root_path in glob.glob(os.path.join(ann_train_root, "*")):
        ann_name = ann_train_root_path.split(os.sep)[-1].split('.')[0]
        img_train_root_path = os.path.join(img_train_root, f"{ann_name}.jpg")
        img_pil = Image.open(img_train_root_path)
        # img = np.array(img_pil)
        w, h = img_pil.size

        lines = []

        root = ET.parse(ann_train_root_path).getroot()
        for tag in root.findall("image/box"):
            label = tag.find("label").text.replace("###", "#").strip("\n").strip("\t").strip("\n")
            gt = list(map(int, tag.find("segs").text.split(",")))
            pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)
            box = get_box(pts, (w, h))
            xn, yn, wn, hn = box
            
            line = f"0 {xn:.4f} {yn:.4f} {wn:.4f} {hn:.4f} " + " ".join(list(map(str, gt))) + f" | {label}"
            assert len(gt) % 2 == 0
            lines.append(line)
        # img = check_ann(img, lines)
        
        img_save_path = os.path.join(img_train_dest, f"{ann_name}.jpg")
        ann_save_path = os.path.join(ann_train_dest, f"{ann_name}.txt")
        with open(ann_save_path, "w") as f:
            f.write("\n".join(lines))
        shutil.copy(img_train_root_path, img_save_path)

    for ann_test_root_path in glob.glob(os.path.join(ann_test_root, "*")):
        ann_name = ann_test_root_path.split(os.sep)[-1].split('.')[0]
        assert ann_name[:3] == "000"
        ann_name = ann_name[3:]
        img_test_root_path = os.path.join(img_test_root, f"{ann_name}.jpg")
        img_pil = Image.open(img_test_root_path)
        img = np.array(img_pil)
        w, h = img_pil.size

        lines = []

        with open(ann_test_root_path, "r") as f:
            root_lines = f.readlines()
            root_lines = [line.strip() for line in root_lines]
        
        for root_line in root_lines:
            line_infos = root_line.split(',')
            text_id = get_text_index(line_infos)
            label = line_infos[text_id:]
            label = ",".join(label)

            assert label[:4] == "####", f"{ann_name} {label}"
            label = label[4:].replace("###", "#")

            gt = list(map(int, line_infos[:text_id]))
            pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)
            box = get_box(pts, (w, h))
            xn, yn, wn, hn = box

            line = f"0 {xn:.4f} {yn:.4f} {wn:.4f} {hn:.4f} " + " ".join(list(map(str, gt))) + f" | {label}"
            assert len(gt) % 2 == 0
            lines.append(line)
        
        img_save_path = os.path.join(img_test_dest, f"{ann_name}.jpg")
        ann_save_path = os.path.join(ann_test_dest, f"{ann_name}.txt")
        with open(ann_save_path, "w") as f:
            f.write("\n".join(lines))
        shutil.copy(img_test_root_path, img_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="data root directory")
    parser.add_argument("--save_dir", required=True,
                        help="save directory")
    parser.add_argument("--normalize_lm", action="store_true",
                        help="enable normalize landmarks")
    args = parser.parse_args()

    main(args)