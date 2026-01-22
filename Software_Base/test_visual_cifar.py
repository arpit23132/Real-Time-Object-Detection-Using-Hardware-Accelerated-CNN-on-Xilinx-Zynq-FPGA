
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt

# ==================== 1. DEFINE CNN ARCHITECTURE ====================
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

# Map IDs to human-readable names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ==================== 2. LOAD DATA & MODEL ====================
def load_cifar_test_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        return batch[b'data'].reshape(-1, 3, 32, 32), np.array(batch[b'labels'])

if __name__ == "__main__":
    cifar_dir = "cifar-10-batches-py"
    test_images, test_labels = load_cifar_test_batch(os.path.join(cifar_dir, "test_batch"))
    test_images_norm = test_images.astype(np.float32) / 255.0

    model = CIFAR_CNN()
    model.load_state_dict(torch.load("cifar_model.pth"))
    model.eval()

    # ==================== 3. RUN TEST & VISUALIZE ====================
    print(f"{'IMAGE':<15} | {'TRUE LABEL':<12} | {'PREDICTION':<12} | {'STATUS'}")
    print("-" * 60)

    found_classes = {}
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.ravel()

    with torch.no_grad():
        for i in range(len(test_labels)):
            label = int(test_labels[i])
            if label not in found_classes:
                # Run Prediction
                input_tensor = torch.FloatTensor(test_images_norm[i:i+1])
                output = model(input_tensor)
                pred_idx = torch.argmax(output, 1).item()
                
                true_name = classes[label]
                pred_name = classes[pred_idx]
                status = "✅" if label == pred_idx else "❌"

                print(f"Label_{label}.txt    | {true_name:<12} | {pred_name:<12} | {status}")

                # Save for C++ verification
                np.savetxt(f"test_image_label_{label}.txt", test_images_norm[i].reshape(-1))

                # Plot the image for your visual check
                # Note: Matplotlib needs (H, W, C) so we transpose from (C, H, W)
                img_to_show = test_images[i].transpose(1, 2, 0)
                axes[label].imshow(img_to_show)
                axes[label].set_title(f"True: {true_name}\nPred: {pred_name}")
                axes[label].axis('off')

                found_classes[label] = True
            if len(found_classes) == 10: break

    plt.tight_layout()
    print("\nDisplaying images... Close the window to finish.")
    plt.show()
