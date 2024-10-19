from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FishRecord:
    trajectory_id: int
    fish_id: int
    species: Tensor
    file_path: str


    def __init__(self, root_path: str, prefix: str, raw_line: str):
        parts = raw_line.split(',')
        index = int(parts[1]) - 1
        self.trajectory_id, self.fish_id = map(int, parts[0].split('_'))
        self.species = torch.nn.functional.one_hot(torch.tensor(index), num_classes=23)
        self.file_path = f"{root_path}/{prefix}_image/{prefix}_{parts[1].zfill(2)}/{prefix}_0{parts[0]}.png"


