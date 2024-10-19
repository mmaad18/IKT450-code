from project.FishRecord import FishRecord
from utils import display_info_project


def main():
    display_info_project()

    root_path = "datasets/Fish_GT"
    label_file_path = root_path + "/class_id.csv"

    with open(label_file_path, 'r') as file:
        next(file)
        data_list = [FishRecord(root_path, "fish", line.strip().split()[0]) for line in file]

    print(len(data_list))


main()


