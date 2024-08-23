import torch
import json
import torchvision.transforms as T
from . import config
from . import utils
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset

class PascalVocDataset(Dataset):
    def __init__(self, set_type: str, normalize=False):
        assert set_type in { "train", "test" }

        self.dataset = VOCDetection(
            root=config.DATA_PATH,
            year="2007",
            image_set=("train" if set_type == "train" else "val"),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.IMAGE_SIZE)
            ])
        )
        self.normalize = normalize
        self.classes = {}

        index = 0
        for data_pair in tqdm(iter(self.dataset), dest="Generating classes dict..."):
            _, label = data_pair
            for bbox_pair in utils.get_bounding_boxes(label):
                name, _ = bbox_pair
                if name not in self.classes:
                    self.classes[name] = index
                    index += 1

    def __getitem__(self, i):
        img_data, label = self.dataset[i]

        # the shape of image data is (channel, height, width)
        grid_size_w = img_data.shape[2] / config.S
        grid_size_h = img_data.shape[0] / config.S

        boxes = {}
        class_names = {}
        depth = config.B * 5 + config.C
        ground_truth = torch.zeros((config.S, config.S, depth))

        for bbox_pair in utils.get_bounding_boxes(label):
            name, coords = bbox_pair
            class_index = self.classes[name]
            assert name in self.classes, f"Unrecognized class {name}"
            x_min, y_min, x_max, y_max = coords

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            col = int(x_mid / grid_size_w)
            row = int(y_mid / grid_size_h)

            assert 0 <= col < config.S, f"The column {col} is invalid when `config.S` is {config.S}"
            assert 0 <= row < config.S, f"The row {row} is invalid when `config.S` is {config.S}"

            cell = (row, col)
            if cell not in class_names.keys() or name == class_names[cell]:
                one_hot = torch.zeros(config.C)
                one_hot[class_index] = 1.0
                ground_truth[row, col, :config.C] = one_hot
                class_names[cell] = name

                bbox_index = boxes.get(cell, 0)
                if bbox_index < config.B:
                    bbox_truth = (
                        (x_min - col * grid_size_w) / config.IMAGE_SIZE[0], # x_min coord relative to grid cell
                        (y_min - col * grid_size_h) / config.IMAGE_SIZE[1], # y_min coord relative to grid cell
                        (x_max - x_min) / config.IMAGE_SIZE[0],             # width
                        (y_max - y_min) / config.IMAGE_SIZE[0],             # height
                        1.0                                                 # confidence
                    )
                    bbox_start = 5 * bbox_index + config.C
                    ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                    boxes[cell] = bbox_index + 1

            return img_data, ground_truth

    def __len__(self):
        return len(self.dataset)
