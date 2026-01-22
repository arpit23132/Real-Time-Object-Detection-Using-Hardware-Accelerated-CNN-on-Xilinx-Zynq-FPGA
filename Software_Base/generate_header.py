import numpy as np

def txt_to_header(txt_file, array_name):
    try:
        data = np.loadtxt(txt_file)
        content = f"const float {array_name}[{len(data)}] = {{\n    "
        # Join numbers with commas
        formatted_data = ", ".join(map(str, data))
        content += formatted_data + "\n};\n\n"
        return content
    except Exception as e:
        print(f"Skipping {txt_file}: {e}")
        return ""

# Map all weight files and all 10 test images
files = {
    "conv1_kernels.txt": "w1_raw",
    "conv1_bias.txt": "b1_raw",
    "conv2_kernels.txt": "w2_raw",
    "conv2_bias.txt": "b2_raw",
    "fc_weights.txt": "wfc_raw",
    "fc_bias.txt": "bfc_raw"
}

# Add images 0-9
for i in range(10):
    files[f"test_image_label_{i}.txt"] = f"img{i}_raw"

with open("data.h", "w") as f:
    f.write("#ifndef DATA_H\n#define DATA_H\n\n")
    for txt, name in files.items():
        print(f"Converting {txt}...")
        f.write(txt_to_header(txt, name))
    f.write("#endif\n")

print("✅ data.h generated successfully!")


