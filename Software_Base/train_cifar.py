import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

# ==================== 1. DEFINE CNN ARCHITECTURE ====================
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        # Padding = 1 keeps spatial dimensions clean
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 32 channels × 8 × 8
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x


# ==================== 2. CIFAR-10 DATA LOADER ====================
def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']

        # Convert to N x 3 x 32 x 32
        data = data.reshape(-1, 3, 32, 32)
        return data, np.array(labels)


if __name__ == "__main__":
    print("Loading CIFAR-10 dataset...")

    cifar_dir = "cifar-10-batches-py"

    train_images = []
    train_labels = []

    # Load all 5 training batches
    for i in range(1, 6):
        data, labels = load_cifar_batch(
            os.path.join(cifar_dir, f"data_batch_{i}")
        )
        train_images.append(data)
        train_labels.append(labels)

    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Normalize to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0

    # ==================== 3. TRAIN / VAL SPLIT ====================
    split_idx = int(0.9 * len(train_images))

    X_train = train_images[:split_idx]
    y_train = train_labels[:split_idx]
    X_val = train_images[split_idx:]
    y_val = train_labels[split_idx:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ==================== 4. TRAIN MODEL ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CIFAR_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    print("\nStarting training...")

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100.0 * train_correct / len(train_dataset)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / len(val_dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

    # ==================== 5. SAVE MODEL & WEIGHTS ====================
    print("\nSaving model and weights...")
    model.eval()
    model = model.cpu()

    # PyTorch model (for Python inference)
    torch.save(model.state_dict(), "cifar_model.pth")
    print("✅ Saved cifar_model.pth")

    # Text weights (for C++ / HLS inference)
    np.savetxt("conv1_kernels.txt", model.conv1.weight.detach().numpy().reshape(-1))
    np.savetxt("conv1_bias.txt", model.conv1.bias.detach().numpy())

    np.savetxt("conv2_kernels.txt", model.conv2.weight.detach().numpy().reshape(-1))
    np.savetxt("conv2_bias.txt", model.conv2.bias.detach().numpy())

    np.savetxt("fc_weights.txt", model.fc.weight.detach().numpy().reshape(-1))
    np.savetxt("fc_bias.txt", model.fc.bias.detach().numpy())

    print("✅ Saved all .txt weights for C++ / FPGA inference")
