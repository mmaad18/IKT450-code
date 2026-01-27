import time
import unittest
from collections import Counter

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.util_4 import get_base_transform, get_test_transform, \
    get_train_transform, get_stats_transform
from project.main_project_utils import images_size, images_size_by_class
from utils import print_time


class MyTestCase(unittest.TestCase):
    training_root_path = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Food_11\\training"
    root_path = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Food_11"

    def test_images_size_histogram(self):
        sizes, paths = images_size(self.training_root_path, "jpg")
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

        self.assertEqual(len(sizes), 9866)


    def test_images_size_scatter_plot(self):
        sizes, paths = images_size(self.training_root_path, "jpg")
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

        self.assertEqual(len(sizes), 9866)


    def test_images_size_scatter_plot_by_class(self):
        # Get image sizes and class information
        sizes, classes, paths = images_size_by_class(self.training_root_path, "jpg")
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

        self.assertEqual(len(sizes), 9866)


    def test_show_class_distribution_log(self):
        sizes, classes, paths = images_size_by_class(self.training_root_path, "jpg")

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

        self.assertEqual(len(sizes), 9866)


    def test_show_class_distribution(self):
        sizes, classes, paths = images_size_by_class(self.training_root_path, "jpg")

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

        self.assertEqual(len(sizes), 9866)


    def test_loading_time(self):
        start = time.perf_counter()

        print_time(start, "Loading training data - START")
        train_data = Food11Dataset("datasets/Food_11", "training", get_base_transform(), get_train_transform())
        print_time(start, "Loading training data complete - COMPLETE")
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        print_time(start, "Creating DataLoader complete - COMPLETE")


    def test_calculate_mean_and_std_dev(self):
        dataset = Food11Dataset(self.root_path, "training", get_base_transform(), get_stats_transform())
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        total_sum = torch.zeros(3)
        total_squared_sum = torch.zeros(3)
        total_pixels = 0

        for images, _ in dataloader:
            batch_size, channels, height, width = images.shape
            batch_pixels = batch_size * height * width
            total_pixels += batch_pixels

            # Sum over batch and spatial dimensions (height and width)
            total_sum += images.sum(dim=[0, 2, 3])
            total_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

        mean = total_sum / total_pixels
        std = torch.sqrt((total_squared_sum / total_pixels) - (mean ** 2))

        print(f"Mean: {mean}")
        print(f"Std: {std}")


    def test_loading_multiple_images(self):
        dataset = Food11Dataset(self.root_path, "training", get_base_transform(), get_train_transform())
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        images, targets = next(iter(dataloader))
        grid_images = torchvision.utils.make_grid(images, nrow=8, padding=10)

        mean = [0.5548, 0.4508, 0.3435]
        std = [0.2651, 0.2674, 0.2747]

        np_image = np.array(grid_images).transpose((1, 2, 0))
        unnorm_image = np_image * std + mean
        plt.figure(figsize=(8, 16))
        plt.imshow(unnorm_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
