import torch
from torch import nn

from project.FishDataset import FishDataset
from project.FishNeuralNetwork import FishNeuralNetwork
from project.main_project_utils import dataset_to_loaders, plot_loss
from utils import display_info_project, load_device
from torchvision import transforms


def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    model.train()

    for batch, (X, T) in enumerate(dataloader):
        X, T = X.to(device), T.to(device)

        Y = model(X)
        loss = loss_fn(Y, T)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # Reset gradients to prevent accumulation


def test_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Prevents PyTorch from calculating and storing gradients
    with torch.no_grad():
        for X, T in dataloader:
            X, T = X.to(device), T.to(device)

            Y = model(X)
            pred_class = (Y > 0.5).float()
            test_loss += loss_fn(Y, T).item()
            correct += (pred_class == T).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss


def main():
    display_info_project()

    device = load_device()
    print(f"Using {device} device")

    model = FishNeuralNetwork().to(device)
    print(model)

    learning_rate = 0.01
    momentum = 0.9
    batch_size = 100
    epochs = 1000

    transform = transforms.Compose([
        transforms.Resize(128),  # Resize the shorter side to 256 and keep the aspect ratio
        transforms.CenterCrop(128),
        transforms.ToTensor()  # Convert the image to a tensor
    ])

    fish_data = FishDataset("datasets/Fish_GT", "fish", transform)
    train_loader, eval_loader, test_loader = dataset_to_loaders(fish_data, batch_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    test_losses = []

    for t in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_loader, model, loss_fn, device)
        test_losses.append(test_loss)

        if t % 1 == 0:
            # Check if CUDA is available
            print(f"CUDA available: {torch.cuda.is_available()}")
            # Check which device your model is on
            print(f"Model device: {next(model.parameters()).device}")
            # Print current memory usage on GPU
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB")

            print(f"Epoch {t}\n-------------------------------")
            print(f"Test Error: {test_loss}\n")

    print("Done!")
    plot_loss("MSE", test_losses, learning_rate, momentum, batch_size)


main()


