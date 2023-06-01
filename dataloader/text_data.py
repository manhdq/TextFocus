import os
import cv2
import numpy as np
from dataloader import TextDataset, TextInstance, pil_load_img
from lxml import etree as ET


def get_absolute_path(p):
    if p.startswith("~"):
        p = os.path.expanduser(p)
    return os.path.abspath(p)


def read_lines(p):
    """Return the text in a file in lines as a list"""
    p = get_absolute_path(p)
    f = open(p, "rU")
    return f.readlines()


class TextData(TextDataset):
    def __init__(
        self,
        data_root,
        is_training=True,
        load_memory=False,
        transform=None,
        ignore_list=None,
    ):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(
            data_root, "Images", "train_images" if is_training else "test_images"
        )
        self.annotation_root = os.path.join(
            data_root, "gt", "train_labels" if is_training else "test_labels"
        )
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = [
            "{}".format(img_name.replace(".jpg", "")) for img_name in self.image_list
        ]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path)
        polygons = []
        for line in lines:
            line = line.split(",")
            pts = np.stack(line[5:-1]).T.astype(np.int32)
            label = line[-1].replace("\n", "")
            polygons.append(TextInstance(pts, "c", label))
        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert c == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        # Read annotation
        if self.is_training:
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_carve_xml(annotation_path)
            pass
        else:
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, "000" + annotation_id)
            polygons = self.parse_carve_txt(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):
        if self.load_memory:
            data = self.datas[item]
        else:
            data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(
                data["image"],
                data["polygons"],
                image_id=data["image_id"],
                image_path=data["image_path"],
            )
        else:
            return self.get_test_data(
                data["image"],
                data["polygons"],
                image_id=data["image_id"],
                image_path=data["image_path"],
            )

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    from utils.augmentation import Augmentation
    from utils import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds)

    trainset = TextData(
        data_root="./data", is_training=True, transform=transform
    )

    for idx in range(0, len(trainset)):
        t0 = time.time()
        (
            img,
            train_mask,
            tr_mask,
            distance_field,
            direction_field,
            weight_matrix,
            ctrl_points,
            proposal_points,
            ignore_tags,
        ) = trainset[idx]
        (
            img,
            train_mask,
            tr_mask,
            distance_field,
            direction_field,
            weight_matrix,
            ctrl_points,
            proposal_points,
            ignore_tags,
        ) = map(
            lambda x: x.cpu().numpy(),
            (
                img,
                train_mask,
                tr_mask,
                distance_field,
                direction_field,
                weight_matrix,
                ctrl_points,
                proposal_points,
                ignore_tags,
            ),
        )

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        boundary_point = ctrl_points[np.where(ignore_tags != 0)[0]]
        for i, bpts in enumerate(boundary_point):
            cv2.drawContours(img, [bpts.astype(np.int32)], -1, (0, 255, 0), 1)
            for j, pp in enumerate(bpts):
                if j == 0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)

            ppts = proposal_points[i]
            cv2.drawContours(img, [ppts.astype(np.int32)], -1, (0, 0, 255), 1)
            for j, pp in enumerate(ppts):
                if j == 0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)
            cv2.imshow("imgs", img)
            cv2.waitKey(0)

    # from util.misc import split_edge_seqence
    # from config_lib.config import config as cfg

    # ret, labels = cv2.connectedComponents(np.array(distance_field >0.35, dtype=np.uint8), connectivity=4)
    # for idx in range(1, ret):
    #     text_mask = labels == idx
    #     ist_id = int(np.sum(text_mask*tr_mask)/np.sum(text_mask))-1
    #     contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     epsilon = 0.007 * cv2.arcLength(contours[0], True)
    #     approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))

    #     pts_num = approx.shape[0]
    #     e_index = [(i, (i + 1) % pts_num) for i in range(pts_num)]
    #     control_points = split_edge_seqence(approx, e_index, cfg.num_points)
    #     control_points = np.array(control_points[:cfg.num_points, :]).astype(np.int32)

    #     cv2.drawContours(img, [ctrl_points[ist_id].astype(np.int32)], -1, (0, 255, 0), 1)
    #     cv2.drawContours(img, [control_points.astype(np.int32)], -1, (0, 0, 255), 1)
    #     for j,  pp in enumerate(control_points):
    #         if j == 0:
    #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
    #         elif j == 1:
    #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
    #         else:
    #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 0), -1)

    #     cv2.imshow('imgs', img)
    #     cv2.waitKey(0)
