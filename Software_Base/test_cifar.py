import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# ==================== 1. DEFINE CNN ARCHITECTURE ====================
# MUST match train_cifar.py EXACTLY
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


# ==================== 2. LOAD CIFAR-10 TEST DATA ====================
def load_cifar_test_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        data = data.reshape(-1, 3, 32, 32)
        return data, np.array(labels)


if __name__ == "__main__":
    # CIFAR-10 class names
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    print("Loading CIFAR-10 test data...")

    cifar_dir = "cifar-10-batches-py"
    test_images, test_labels = load_cifar_test_batch(
        os.path.join(cifar_dir, "test_batch")
    )

    # Normalize
    test_images = test_images.astype(np.float32) / 255.0

    # ==================== 3. LOAD TRAINED MODEL ====================
    print("Loading trained CIFAR model...")
    device = torch.device("cpu")  # CPU is enough for verification
    model = CIFAR_CNN().to(device)

    try:
        model.load_state_dict(torch.load("cifar_model.pth"))
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: cifar_model.pth not found. Run train_cifar.py first.")
        exit()

    model.eval()

    # ==================== 4. GENERATE TEST FILES (ONE PER CLASS) ====================
    print("\nFinding one example for each CIFAR-10 class (0–9)...")

    found_classes = {}

    with torch.no_grad():
        for i in range(len(test_labels)):
            label = int(test_labels[i])

            if label not in found_classes:
                filename = f"test_image_label_{label}.txt"

                # A. Save raw image for C++ (3 x 32 x 32)
                img_data = test_images[i]
                np.savetxt(filename, img_data.reshape(-1))

                # B. Python inference check
                input_tensor = torch.FloatTensor(test_images[i:i+1])
                output = model(input_tensor)
                predicted = torch.argmax(output, 1).item()

                # Enhanced output with class names
                status = "[SUCCESS]" if predicted == label else "[FAIL]"
                print(
                    f"✅ Saved {filename} | "
                    f"True: {class_names[label]:12s} | "
                    f"Pred: {class_names[predicted]:12s} {status}"
                )

                found_classes[label] = True

            if len(found_classes) == 10:
                break

    print("\n" + "="*50)
    print("All CIFAR-10 test files generated. Ready for C++ inference.")
    print("="*50)
