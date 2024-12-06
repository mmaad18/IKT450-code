import os
import unittest
from collections import Counter

import torch
from torchvision.io import decode_image
from torchvision.transforms import v2
from PIL import Image

import numpy as np

from project.main_project_utils import images_size, path_to_fish_id, images_size_by_class, crop_black_borders

from matplotlib import pyplot as plt


class MainProjectTest(unittest.TestCase):
    root_path = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Fish_GT\\image_cropped"


    def test_crop_black_borders(self):
        file_path = os.path.join(self.root_path, "fish_08\\fish_003269959708_05273.png")

        cropped_image = crop_black_borders(file_path)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image", fontsize=20)
        plt.imshow(Image.open(file_path))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Cropped Image", fontsize=20)
        plt.imshow(cropped_image)
        plt.axis("off")

        plt.show()




    def test_images_size(self):
        sizes, paths = images_size(self.root_path)

        max_height_idx = np.argmax(sizes[:, 0].astype(int))
        max_width_idx = np.argmax(sizes[:, 1].astype(int))
        min_height_idx = np.argmin(sizes[:, 0].astype(int))
        min_width_idx = np.argmin(sizes[:, 1].astype(int))

        max_height_size = sizes[max_height_idx]
        max_height_path = paths[max_height_idx]
        max_height_fish_id = path_to_fish_id(max_height_path)

        max_width_size = sizes[max_width_idx]
        max_width_path = paths[max_width_idx]
        max_width_fish_id = path_to_fish_id(max_width_path)

        min_height_size = sizes[min_height_idx]
        min_height_path = paths[min_height_idx]
        min_height_fish_id = path_to_fish_id(min_height_path)

        min_width_size = sizes[min_width_idx]
        min_width_path = paths[min_width_idx]
        min_width_fish_id = path_to_fish_id(min_width_path)

        print("min_height_path: ", min_height_path)
        print("min_width_path: ", min_width_path)
        print("max_height_path: ", max_height_path)
        print("max_width_path: ", max_width_path)

        self.assertEqual(len(sizes), 27370)
        self.assertTrue(np.array_equal(max_height_size, np.array([428, 401])))
        self.assertTrue(np.array_equal(max_width_size, np.array([428, 401])))
        self.assertTrue(np.array_equal(min_height_size, np.array([25, 27])))
        self.assertTrue(np.array_equal(min_width_size, np.array([25, 27])))

        self.assertEqual(max_height_fish_id, 5273)
        self.assertEqual(max_width_fish_id, 5273)
        self.assertEqual(min_height_fish_id, 3992)
        self.assertEqual(min_width_fish_id, 3992)


    def test_images_size_histogram(self):
        sizes, paths = images_size(self.root_path)
        widths = sizes[:, 0]
        heights = sizes[:, 1]

        plt.figure(figsize=(10, 5))

        # Histogram for widths
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=100, color='blue')
        plt.title('Distribution of Image Widths', fontsize=20)
        plt.xlabel('Width', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tick_params(labelsize=16)

        # Histogram for heights
        plt.subplot(1, 2, 2)
        plt.hist(heights, bins=100, color='green')
        plt.title('Distribution of Image Heights', fontsize=20)
        plt.xlabel('Height', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tick_params(labelsize=14)

        plt.tight_layout()
        plt.show()

        self.assertEqual(len(sizes), 27370)


    def test_images_size_scatter_plot(self):
        sizes, paths = images_size(self.root_path)
        widths = sizes[:, 0]
        heights = sizes[:, 1]

        # Scatter plot for image sizes (Width vs Height)
        plt.figure(figsize=(8, 8))
        plt.scatter(widths, heights, color='purple', alpha=0.3)
        plt.title('Image Sizes (Width vs Height)', fontsize=20)
        plt.xlabel('Width', fontsize=16)
        plt.ylabel('Height', fontsize=16)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.show()

        self.assertEqual(len(sizes), 27370)


    def test_images_size_scatter_plot_by_class(self):
        # Get image sizes and class information
        sizes, classes, paths = images_size_by_class(self.root_path)
        widths = sizes[:, 0]
        heights = sizes[:, 1]

        # Assign a unique color to each class
        unique_classes = sorted(set(classes))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))  # Use colormap with enough distinct colors
        class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

        # Scatter plot for image sizes (Width vs Height) by class
        plt.figure(figsize=(8, 8))
        for cls in unique_classes:
            cls_indices = [i for i, c in enumerate(classes) if c == cls]
            cls_widths = widths[cls_indices]
            cls_heights = heights[cls_indices]
            plt.scatter(cls_widths, cls_heights, color=class_to_color[cls], label=cls, alpha=0.5, edgecolors="w", s=20)

        plt.title('Image Sizes (Width vs Height) by Class', fontsize=20)
        plt.xlabel('Width', fontsize=16)
        plt.ylabel('Height', fontsize=16)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.show()

        self.assertEqual(len(sizes), 27370)


    def test_show_class_distribution_log(self):
        sizes, classes, paths = images_size_by_class(self.root_path)

        # Count the number of images per class
        class_counts = Counter(classes)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        # Plot the distribution
        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts, color="skyblue", edgecolor="black")
        plt.title("Number of Images Per Class", fontsize=20)
        plt.xlabel("Class", fontsize=16)
        plt.ylabel("Number of Images", fontsize=16)
        plt.tick_params(labelsize=14)
        plt.yscale("log")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis='y', which="both", linestyle="--", linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()
        plt.show()

        self.assertEqual(len(sizes), 27370)


    def test_show_class_distribution(self):
        sizes, classes, paths = images_size_by_class(self.root_path)

        # Count the number of images per class
        class_counts = Counter(classes)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        # Sort classes and counts for proper order
        sorted_classes = sorted(zip(classes, counts), key=lambda x: x[0])
        classes, counts = zip(*sorted_classes)

        # Plot the distribution as a bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts, color="skyblue", edgecolor="black")

        # Add counts on top of each bar
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X-coordinate: center of the bar
                height,  # Y-coordinate: top of the bar
                f'{count}',  # Text: the count
                ha='center', va='bottom', fontsize=11  # Align center and bottom of the bar
            )

        # Add title and labels
        plt.title("Number of Images Per Class", fontsize=20)
        plt.xlabel("Class", fontsize=16)
        plt.ylabel("Number of Images", fontsize=16)
        plt.tick_params(labelsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        self.assertEqual(len(sizes), 27370)


    def test_image_transform(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        #file_path = os.path.join(self.root_path, "fish_07\\fish_000017689596_03992.png")
        file_path = os.path.join(self.root_path, "fish_08\\fish_003269959708_05273.png")

        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=(64, 64)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=.25, hue=.15),
            v2.GaussianNoise(mean=0.0 , sigma=0.05),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.)),
        ])

        image = decode_image(file_path)
        print(f"{type(image) = }, {image.dtype = }, {image.shape = }")
        tensor = transforms(image)
        transformed_image = tensor.permute(1, 2, 0).numpy()
        image = image.permute(1, 2, 0).numpy()

        # Plot the original and transformed images side by side
        plt.figure(figsize=(8, 4))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image", fontsize=20)
        plt.axis("off")

        # Transformed image
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_image)
        plt.title("Transformed Image", fontsize=20)
        plt.axis("off")

        # Show the plots
        plt.tight_layout()
        plt.show()





if __name__ == '__main__':
    unittest.main()

