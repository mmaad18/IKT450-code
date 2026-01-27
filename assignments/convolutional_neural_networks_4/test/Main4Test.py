import os
import unittest
from pathlib import Path

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.io import decode_image
from torchvision.transforms import v2

from assignments.convolutional_neural_networks_4.networks.ResNet18 import ResNet18
from assignments.convolutional_neural_networks_4.networks.ResNet34 import ResNet34
from assignments.convolutional_neural_networks_4.util_4 import get_test_transform, get_train_transform, \
    get_base_transform
from utils import logs_path, load_plotly_webbrowser, load_device, load_model


class Main4Test(unittest.TestCase):
    training_root_path = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Food_11\\training"
    root_path = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Food_11"


    def test_image_transform(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        file_paths = [
            os.path.join(self.root_path, "validation\\Bread\\3.jpg"),
            os.path.join(self.training_root_path, "Dairy product\\0.jpg"),
            os.path.join(self.root_path, "validation\\Dessert\\24.jpg"),
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

        # transform = v2.Compose([
        #     v2.ToImage(),
        #     v2.Resize((96, 96)),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.Normalize(
        #         mean=[0.5607, 0.4520, 0.3385],
        #         std=[0.2598, 0.2625, 0.2692],
        #     ),
        # ])

        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop((96, 96)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            v2.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            v2.Normalize(
                mean=[0.5607, 0.4520, 0.3385],
                std=[0.2598, 0.2625, 0.2692],
            ),
        ])

        # transform = v2.Compose([
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.RandomResizedCrop(size=96, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        #     v2.RandomHorizontalFlip(p=0.5),
        #     #v2.RandomVerticalFlip(p=0.5),
        #     v2.RandomErasing(p=0.99, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        #     v2.RandomRotation(degrees=15),
        #     v2.RandomErasing(p=0.99, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        #     v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        #     #v2.RandomChoice([
        #     #    v2.GaussianNoise(mean=0.0, sigma=0.05),
        #     #    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0))
        #     #]),
        #     v2.Normalize(mean=[0.5607, 0.4520, 0.3385], std=[0.2598, 0.2625, 0.2692])
        # ])

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


    def test_print_transform_info(self):
        transform = get_train_transform()
        print(transform)


    def test_load_plotly_to_webbrowser(self):
        run_path = logs_path("A4_Res18_260106_021448")

        path_list = [
            #run_path / "confusion_matrix.html",
            #run_path / "confusion_matrix_aggregate.html",
            #run_path / "confusion_matrix_test.html",
            run_path / "plotly_metrics.html",
            #run_path / "plotly_metrics_aggregate.html"
        ]

        for path in path_list:
            load_plotly_webbrowser(path)


    def test_load_model(self):
        category_list = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
            'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]

        run_id = "A4_Res18_260105_024346"
        #image_path = os.path.join(self.root_path, f"evaluation\\{category_list[0]}\\2.jpg")
        image_path = Path("assignments/convolutional_neural_networks_4/test/images/spaghetti_1.png")
        image = Image.open(image_path).convert("RGB")
        transform = v2.Compose([get_base_transform(), get_test_transform()])
        x_transform = transform(image)
        x = x_transform.unsqueeze(0)

        device = load_device()
        model = ResNet18(device)
        model = load_model(run_id, model, device)
        model.eval()

        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()

        labels = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
            'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]

        print("Predicted class:", labels[pred_idx])
        print("Confidence:", probs[0, pred_idx].item())

        transformed_image = x_transform.permute(1, 2, 0).numpy()
        plt.imshow(transformed_image)
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    unittest.main()

