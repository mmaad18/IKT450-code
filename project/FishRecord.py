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
        idx = int(parts[1]) - 1
        trajectory_str, fish_str = parts[0].split('_')
        self.trajectory_id = int(trajectory_str)
        self.fish_id = int(fish_str)
        self.species = torch.nn.functional.one_hot(torch.tensor(idx), num_classes=23).float()
        self.file_path = f"{root_path}/{prefix}_image/{prefix}_{parts[1].zfill(2)}/{prefix}_{trajectory_str.zfill(12)}_{fish_str}.png"


