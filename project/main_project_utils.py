import os

import numpy as np
from PIL import Image, ImageChops


def images_size(root_path: str, file_type="png"):
    sizes = []
    paths = []
    file_ending = "." + file_type

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(file_ending):
                file_path = os.path.join(folder_path, file)
                with Image.open(file_path) as img:
                    sizes.append(img.size)
                    paths.append(file_path)

    return np.array(sizes), paths


def images_size_by_class(root_path: str, file_type="png"):
    sizes = []
    classes = []
    paths = []
    file_ending = "." + file_type

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(file_ending):
                    file_path = os.path.join(folder_path, file)
                    with Image.open(file_path) as img:
                        sizes.append(img.size)
                        classes.append(folder)
                        paths.append(file_path)

    return np.array(sizes), classes, paths


def path_to_fish_id(path: str):
    file_name = path.split('\\')[-1]
    return int(file_name.split('_')[-1].split('.')[0])


def crop_black_borders(image_path: str, counter: int):
    image = Image.open(image_path)

    # Get the bounding box of the non-black region
    bounding_box = Image.new(image.mode, image.size, (0, 0, 0))
    diff = ImageChops.difference(image, bounding_box)
    bbox = diff.getbbox()

    if bbox:
        cropped_img = image.crop(bbox)
        return cropped_img, (counter + 1)
    else:
        return image, counter


def crop_borders_all_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        output_folder_path = os.path.join(output_dir, folder)

        if os.path.isdir(folder_path):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    file_path = os.path.join(folder_path, file)
                    output_file_path = os.path.join(output_folder_path, file)

                    cropped_image, counter = crop_black_borders(file_path, counter)
                    cropped_image.save(output_file_path)

                    if counter % 10 == 0:
                        print(f"{counter} images cropped")

    print(f"{counter} images cropped")

