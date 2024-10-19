from dataclasses import dataclass


@dataclass
class FishRecord:
    trajectory_id: int
    fish_id: int
    species: int
    file_path: str

    def __init__(self, root_path: str, prefix: str, raw_line: str):
        parts = raw_line.split(',')
        self.trajectory_id, self.fish_id = map(int, parts[0].split('_'))
        self.species = int(parts[1])
        self.file_path = f"{root_path}/{prefix}_image/{prefix}_{parts[1].zfill(2)}/{prefix}_0{parts[0]}.png"


