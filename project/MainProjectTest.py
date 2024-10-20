import unittest
import numpy as np

from project.main_project_utils import images_size, path_to_fish_id

from matplotlib import pyplot as plt


class MainProjectTest(unittest.TestCase):
    def test_images_size(self):
        sizes, paths = images_size("C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Fish_GT\\fish_image")

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
        sizes, paths = images_size("C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Fish_GT\\fish_image")
        widths = sizes[:, 0]
        heights = sizes[:, 1]

        plt.figure(figsize=(10, 5))

        # Histogram for widths
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=100, color='blue')
        plt.title('Distribution of Image Widths')
        plt.xlabel('Width')
        plt.ylabel('Frequency')

        # Histogram for heights
        plt.subplot(1, 2, 2)
        plt.hist(heights, bins=100, color='green')
        plt.title('Distribution of Image Heights')
        plt.xlabel('Height')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        self.assertEqual(len(sizes), 27370)


    def test_images_size_scatter_plot(self):
        sizes, paths = images_size("C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Fish_GT\\fish_image")
        widths = sizes[:, 0]
        heights = sizes[:, 1]

        # Scatter plot for image sizes (Width vs Height)
        plt.figure(figsize=(6, 6))
        plt.scatter(widths, heights, color='purple', alpha=0.3)
        plt.title('Image Sizes (Width vs Height)')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.grid(True)
        plt.show()

        self.assertEqual(len(sizes), 27370)


if __name__ == '__main__':
    unittest.main()

