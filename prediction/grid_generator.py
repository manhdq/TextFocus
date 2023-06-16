'''
Grid Generator used to preprocess image before demo or inference
'''

import math

import cv2
import numpy as np

from .prediction_utils import find_new_size


class GridGenerator():
    '''
    Grid Generator used to preprocess image before inference or inference
    '''
    def __init__(self,
                 max_valid_size,
                 grid_threshold,
                 overlap_ratio,
                 base_scale_down,
                 valid_range,
                 interpolation,
                 max_chip_size):
        self.max_valid_size = max_valid_size
        self.grid_threshold = grid_threshold
        self.overlap_ratio = overlap_ratio
        self.base_scale_down = base_scale_down
        self.valid_range = valid_range
        self.interpolation = interpolation
        self.max_chip_size = max_chip_size

    def gen_from_ori_img(self, image):
        '''
        Generate grids from original image
        '''
        self.image_height, self.image_width = image.shape[:2]
        self.max_img_size = max(self.image_width, self.image_height)
        if self.max_img_size >= self.max_valid_size:
            return None, self.base_scale_down
        return self._gen_grids(image)

    def _gen_grids(self, image):
        if self.max_img_size >= self.grid_threshold:
            if self.max_img_size == self.image_width:  # Vertical crop
                new_width = math.ceil(self._find_crop_size(self.image_width))
                tile_1 = image[:, :new_width, :].copy()
                tile_2 = image[:, -new_width:, :].copy()

                tile_1, tile_2, new_scale_down = self._resize_tiles(tile_1, tile_2)

                scaled_down_tiles = [
                    {
                        "image": tile_1,
                        "prev_left_shift": 0,
                        "prev_top_shift": 0
                    },
                    {
                        "image": tile_2,
                        "prev_left_shift": self.image_width - new_width,
                        "prev_top_shift": 0
                    }
                ]
            else: # Horizontal crop
                new_height = math.ceil(self._find_crop_size(self.image_height))
                tile_1 = image[:new_height, :, :].copy()
                tile_2 = image[-new_height:, :, :].copy()

                tile_1, tile_2, new_scale_down = self._resize_tiles(tile_1, tile_2)

                scaled_down_tiles = [
                    {
                        'image': tile_1,
                        'prev_left_shift': 0,
                        'prev_top_shift': 0
                    },
                    {
                        'image': tile_2,
                        'prev_left_shift': 0,
                        'prev_top_shift': self.image_height - new_height
                    }
                ]
        else:
            new_size, new_scale_down = find_new_size(image_width=self.image_width,
                                                    image_height=self.image_height,
                                                    valid_range=self.valid_range,
                                                    base_scale_down=self.base_scale_down)

            if new_scale_down != 1:
                scaled_down_image = cv2.resize(image, new_size, interpolation=self.interpolation)
            else:
                scaled_down_image = image.copy()
            scaled_down_tiles = [{
                'image': scaled_down_image,
                'prev_left_shift': 0,
                'prev_top_shift': 0
            }]

        return scaled_down_tiles, new_scale_down

    def _find_crop_size(self, size):
        return (1 + self.overlap_ratio) * (size / 2)

    def _resize_tiles(self, tile_1, tile_2):
        tile_height, tile_width = tile_1.shape[:2]

        new_size, new_scale_down = find_new_size(image_width=tile_width,
                                                image_height=tile_height,
                                                valid_range=self.valid_range,
                                                base_scale_down=self.base_scale_down)

        if new_scale_down != 1:
            tile_1 = cv2.resize(tile_1, new_size, interpolation=self.interpolation)
            tile_2 = cv2.resize(tile_2, new_size, interpolation=self.interpolation)

        return tile_1, tile_2, new_scale_down

    def gen_from_chip(self, chip_coord, rank):
        """
        Generate grids from chip
        """
        x1, y1, x2 ,y2 = chip_coord
        w = x2 - x1
        h = y2 - y1
        max_size = max(w, h)
        if max_size < self.max_chip_size or rank == 0:
            return [chip_coord]

        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        return np.asarray([[x1, y1, mid_x + margin_x, mid_y + margin_y],
                           [mid_x - margin_x, y1, x2, mid_y + margin_y],
                           [x1, mid_y + margin_y, mid_x - margin_x, y2],
                           [mid_x - margin_x, mid_y - margin_y, x2, y2]])