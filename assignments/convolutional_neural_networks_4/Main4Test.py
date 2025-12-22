import os
import unittest
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2
from PIL import Image

import numpy as np

from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from project.main_project_utils import images_size, path_to_fish_id, images_size_by_class, crop_black_borders

from matplotlib import pyplot as plt


class Main4Test(unittest.TestCase):
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


    def test_image_transform(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        file_paths = [
            os.path.join(self.training_root_path, "Bread\\0.jpg"),
            os.path.join(self.training_root_path, "Dairy product\\0.jpg"),
            os.path.join(self.training_root_path, "Dessert\\0.jpg"),
            os.path.join(self.training_root_path, "Egg\\0.jpg"),
            os.path.join(self.training_root_path, "Fried food\\0.jpg"),
            os.path.join(self.training_root_path, "Meat\\0.jpg"),
            os.path.join(self.training_root_path, "Noodles-Pasta\\0.jpg"),
            os.path.join(self.training_root_path, "Rice\\0.jpg"),
            os.path.join(self.training_root_path, "Seafood\\0.jpg"),
            os.path.join(self.training_root_path, "Soup\\0.jpg"),
            os.path.join(self.training_root_path, "Vegetable-Fruit\\0.jpg")
        ]

        file_path = file_paths[0]

        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=96, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            v2.RandomHorizontalFlip(p=0.5),
            #v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            #v2.RandomChoice([
            #    v2.GaussianNoise(mean=0.0, sigma=0.05),
            #    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0))
            #]),
            v2.Normalize(mean=[0.5607, 0.4520, 0.3385], std=[0.2598, 0.2625, 0.2692])
        ])

        image = decode_image(file_path)
        print(f"{type(image) = }, {image.dtype = }, {image.shape = }")
        tensor = transform(image)
        transformed_image = tensor.permute(1, 2, 0).numpy()
        image = image.permute(1, 2, 0).numpy()

        # Plot the original and transformed images side by side
        plt.figure(figsize=(8, 4))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Original ({image.shape[0]}, {image.shape[1]})", fontsize=20)
        plt.axis("off")

        # Transformed image
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_image)
        plt.title(f"Transformed ({transformed_image.shape[0]}, {transformed_image.shape[1]})", fontsize=20)
        plt.axis("off")

        # Show the plots
        plt.tight_layout()
        plt.show()


    def test_calculate_mean_and_std_dev(self):
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=96),
            v2.CenterCrop(size=96),
            v2.ToTensor()
        ])

        # Load the dataset with minimal preprocessing
        dataset = Food11Dataset(self.root_path, "training", transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        # Initialize sums and squared sums
        mean = torch.zeros(3)
        std = torch.zeros(3)

        for images, _ in dataloader:
            # Sum over batch and spatial dimensions (height and width)
            mean += images.mean(dim=[0, 2, 3])
            std += images.std(dim=[0, 2, 3])

        # Divide by the number of batches to get mean and std
        mean /= len(dataloader)
        std /= len(dataloader)

        print(f"Mean: {mean}")
        print(f"Std: {std}")


if __name__ == '__main__':
    unittest.main()

