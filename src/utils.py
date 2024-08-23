import os
import json
import unittest
import torch
from typing import Tuple
from . import config

def calc_iou(prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    '''
    prediction.shape: (config.BATCH_SIZE, config.S, config.S, config.C + config.B * 5)
    ground_truth.shape: (config.BATCH_SIZE, config.S, config.S, config.C + config.B * 5)
    '''
    p_rect_min, p_rect_max = get_coords(prediction)                                 # shape: (BATCH_SIZE, S, S, B, 2)
    gt_rect_min, gt_rect_max = get_coords(ground_truth)                             # shape: (BATCH_SIZE, S, S, B, 2)

    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    overlap_rect_min = torch.max(
        p_rect_min.unsqueeze(4).expand(coords_join_size),                           # shape: (BATCH_SIZE, S, S, B, 1, 2) -> (BATCH_SIZE, S, S, B, B, 2)
        gt_rect_min.unsqueeze(3).expand(coords_join_size)                           # shape: (BATCH_SIZE, S, S, 1, B, 2) -> (BATCH_SIZE, S, S, B, B, 2)
    )
    overlap_rect_max = torch.min(
        p_rect_max.unsqueeze(4).expand(coords_join_size),                           # shape: (BATCH_SIZE, S, S, B, 1, 2) -> (BATCH_SIZE, S, S, B, B, 2)
        gt_rect_max.unsqueeze(3).expand(coords_join_size)                           # shape: (BATCH_SIZE, S, S, 1, B, 2) -> (BATCH_SIZE, S, S, B, B, 2)
    )

    intersection_rect = torch.clamp(overlap_rect_max - overlap_rect_min, min=0.0)
    intersection_area = intersection_rect[..., 0] * intersection_rect[..., 1]       # shape: (BATCH_SIZE, S, S, B, B)

    p_area = get_bbox_attr_from_nn_result_data(prediction, 2) \
             * get_bbox_attr_from_nn_result_data(prediction, 3)
    p_area = p_area.unsqueeze(4).expand_as(intersection_area)

    gt_area = get_bbox_attr_from_nn_result_data(ground_truth, 2) \
              * get_bbox_attr_from_nn_result_data(ground_truth, 3)
    gt_area = gt_area.unsqueeze(3).expand_as(intersection_area)

    union_area = p_area + gt_area - intersection_area
    is_union_area_zero = (union_area == 0.0)
    union_area[is_union_area_zero] = config.EPSILON

    return intersection_area / union_area

def get_coords(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    data.shape: (config.BATCH_SIZE, config.S, config.S, config.C + config.B * 5)
    '''
    bboxs_x_min = get_bbox_attr_from_nn_result_data(data, 0)                                                # shape: (BATCH_SIZE, S, S, B)
    bboxs_y_min = get_bbox_attr_from_nn_result_data(data, 1)                                                # shape: (BATCH_SIZE, S, S, B)
    bboxs_width = get_bbox_attr_from_nn_result_data(data, 2)                                                # shape: (BATCH_SIZE, S, S, B)
    bboxs_height = get_bbox_attr_from_nn_result_data(data, 3)                                               # shape: (BATCH_SIZE, S, S, B)

    bboxs_x_max = bboxs_x_min + bboxs_width                                                                 # shape: (BATCH_SIZE, S, S, B)
    bboxs_y_max = bboxs_y_min + bboxs_height                                                                # shape: (BATCH_SIZE, S, S, B)

    return torch.stack((bboxs_x_min, bboxs_y_min), dim=4), torch.stack((bboxs_x_max, bboxs_y_max), dim=4)   # shape of each element of the tuple: (BATCH_SIZE, S, S, B, 2)

def get_bbox_attr_from_nn_result_data(data: torch.Tensor, i: int) -> torch.Tensor:
    '''
    Get the ith attr for each bounding box in result data of NN.
    '''
    offset = config.C + i
    return data[..., offset::5]

# def calc_iou(box_a: torch.Tensor, box_b: torch.Tensor):
#     box_a_tl = (box_a[0], box_a[1])
#     box_a_br = (box_a[0] + box_a[2], box_a[1] + box_a[3])
#
#     box_b_tl = (box_b[0], box_b[1])
#     box_b_br = (box_b[0] + box_b[2], box_b[1] + box_b[3])
#
#     overlap_tl = (torch.max(box_a_tl[0], box_b_tl[0]), torch.max(box_a_tl[1], box_b_tl[1]))
#     overlap_br = (torch.min(box_a_br[0], box_b_br[0]), torch.min(box_a_br[1], box_b_br[1]))
#
#     overlap_area = torch.max((overlap_br[0] - overlap_tl[0]) * (overlap_br[1] - overlap_tl[1]), torch.Tensor([0.]))
#
#     union_area = (box_a[2] * box_a[3]) + (box_b[2] * box_b[3]) - overlap_area
#
#     return overlap_area / union_area


# def save_class_dict(obj):
#     folder = os.path.dirname(config.CLASSES_PATH)
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     with open(config.CLASSES_PATH, "w") as file:
#         json.dump(obj, file, indent=2)
#
#
# def load_class_dict():
#     if os.path.exists(config.CLASSES_PATH):
#         with open(config.CLASSES_PATH, "r") as file:
#             return json.load(file)
#     new_dict = {}
#     save_class_dict(new_dict)
#     return new_dict


def get_bounding_boxes(label):
    size = label["annotation"]["size"]
    w, h = int(size["width"]), int(size["height"])
    x_scale = config.IMAGE_SIZE[0] / w
    y_scale = config.IMAGE_SIZE[1] / h
    boxes = []
    objects = label["annotation"]["object"]
    for obj in objects:
        box = obj["bndbox"]
        coords = (
            int(int(box["xmin"]) * x_scale),
            int(int(box["ymin"]) * y_scale),
            int(int(box["xmax"]) * x_scale),
            int(int(box["ymax"]) * y_scale)
        )
        name = obj["name"]
        boxes.append((name, coords))

    return boxes


class UtilsUnitTests(unittest.TestCase):
    def test_calc_iou(self):
        a = torch.tensor([1, 1, 2, 2, 0.1])
        b = torch.tensor([2, 2, 3, 3, 0.2])

        iou = calc_iou(a, b)
        self.assertEqual(iou, 1 / 12)
