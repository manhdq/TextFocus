import numpy as np
import cv2


def xywh2xyxy(box, img_size):
    w, h = img_size
    new_box = np.zeros_like(box)

    new_box[0] = (box[0] - box[2] / 2) * w
    new_box[1] = (box[1] - box[3] / 2) * h
    new_box[2] = (box[0] + box[2] / 2) * w
    new_box[3] = (box[1] + box[3] / 2) * h
    return new_box


def check_ann(image, lines):
    image_show = image.copy()
    h, w = image_show.shape[:2]
    image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min()) * 255
    image_show = np.ascontiguousarray(image_show[:, :, ::-1]).astype(np.uint8)

    box_list = []
    kpts_list = []
    text_list = []
    is_valids = []
    for line in lines:
        ann_infos, text = line.strip().split(" | ")
        ann_infos = ann_infos.strip().split()
        text = text.strip()
        # print(text)

        cls = int(ann_infos[0])
        is_valids.append(int(ann_infos[1]))
        box = np.array(list(map(float, ann_infos[2:6])))
        lms = list(map(float, ann_infos[6:]))

        box = xywh2xyxy(box, (w, h))
        pts = np.stack([lms[0::2], lms[1::2]]).T.astype(np.int32)
        
        box_list.append(box)
        kpts_list.append(pts)
        text_list.append(text)

    for box, text in zip(box_list, text_list):
        box = box.astype(int)
        color = (255, 0, 0)
        image_show = cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]),
                                color, 1)
    
    for point, is_valid in zip(kpts_list, is_valids):
        boundary_color = (0, 255, 0) if is_valid else (0, 0, 255)
        image_show = cv2.polylines(image_show,
                                [point.astype(int)], True, boundary_color, 2)

    return image_show


if __name__ == "__main__":
    from PIL import Image

    img_paths = ["/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/Images/chip_for_train/0323_1.jpg",
                "/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/Images/chip_for_train/0327_1.jpg",
                "/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/Images/chip_for_train/0327_2.jpg"]

    ann_paths = ["/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/gt/chip_for_train/0323_1.txt",
                "/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/gt/chip_for_train/0327_1.txt",
                "/home/ubuntu/Documents/working/pixtaVN/RA/TextBPN++/data/CTW1500/yolo/gt/chip_for_train/0327_2.txt"]

    for i, (img_path, ann_path) in enumerate(zip(img_paths, ann_paths)):
        img = np.array(Image.open(img_path).convert("RGB"))

        with open(ann_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        img = check_ann(img, lines)

        cv2.imwrite(f"test_{i}.jpg", img)