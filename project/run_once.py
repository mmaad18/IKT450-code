import os

from project.main_project_utils import crop_borders_all_images


def run_once():
    input_dir = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Fish_GT\\fish_image"
    output_dir = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Fish_GT\\image_cropped"

    crop_borders_all_images(input_dir, output_dir)


run_once()

