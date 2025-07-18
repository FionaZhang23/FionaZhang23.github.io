import numpy as np
import os
import matplotlib.pyplot as plt

# Load image and label data
data_path = "/deac/csc/classes/csc373/zhanx223/assignment_4/output"
scan_fall_2019 = np.load(os.path.join(data_path, "filtered_scans.npy"))  # (N, H, W) or (N, H, W, C)
labels_fall_2019 = np.load(os.path.join(data_path, "filtered_labels.npy"))  # (N, H, W, 4) (One-hot encoding)

# Convert one-hot labels to categorical format (0,1,2,3)
labels_fall_2019 = np.argmax(labels_fall_2019, axis=-1)  # Shape: (N, H, W)

# Check if the images are grayscale or RGB
if len(scan_fall_2019.shape) == 4 and scan_fall_2019.shape[-1] == 3:
    color_mode = "RGB"
elif len(scan_fall_2019.shape) == 3:
    color_mode = "Grayscale"
else:
    color_mode = "Unknown"

# Compute basic statistics
num_images = scan_fall_2019.shape[0]
image_shape = scan_fall_2019.shape[1:]  # Height, Width (and Channels if applicable)
pixel_mean = np.mean(scan_fall_2019)
pixel_std = np.std(scan_fall_2019)
pixel_min = np.min(scan_fall_2019)
pixel_max = np.max(scan_fall_2019)

# Compute label distribution
label_counts = np.bincount(labels_fall_2019.flatten(), minlength=4)

# Save label distribution histogram
plt.figure(figsize=(8, 5))
plt.bar([0, 1, 2, 3], label_counts, color=["red", "blue", "green", "gray"])
plt.xlabel("Label")
plt.ylabel("Pixel Count")
plt.title("Label Distribution in Training Dataset")
plt.xticks([0, 1, 2, 3])
plt.savefig("/deac/csc/classes/csc373/zhanx223/assignment_4/output/label_distribution.png")
plt.close()

# Generate color (pixel intensity) distribution
plt.figure(figsize=(8, 5))
plt.hist(scan_fall_2019.flatten(), bins=50, color="purple", alpha=0.7, edgecolor="black")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Pixel Intensity Distribution")
plt.savefig("/deac/csc/classes/csc373/zhanx223/assignment_4/output/color_distribution.png")
plt.close()

if scan_fall_2019.ndim == 4 and scan_fall_2019.shape[-1] == 1:
    scan_fall_2019 = np.squeeze(scan_fall_2019, axis=-1)

# Select a few images to display
num_rows = 10
num_cols = 5
num_samples = num_rows * num_cols
sample_indices = np.linspace(0, scan_fall_2019.shape[0] - 1, num_samples, dtype=int)

# Create a figure for displaying images
fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(20, 30))

for i, idx in enumerate(sample_indices):
    row, col = divmod(i, num_cols)

    # Display the scan image
    axes[row, col * 2].imshow(scan_fall_2019[idx], cmap="gray")
    axes[row, col * 2].set_title(f"Scan {idx}", fontsize=8)
    axes[row, col * 2].axis("off")

    # Display the corresponding label
    axes[row, col * 2 + 1].imshow(labels_fall_2019[idx], cmap="jet", alpha=0.6)
    axes[row, col * 2 + 1].set_title(f"Label {idx}", fontsize=8)
    axes[row, col * 2 + 1].axis("off")

plt.suptitle("Sample Radiology Scans with Corresponding Labels (50 Images)", fontsize=16)
plt.tight_layout()
plt.savefig("/deac/csc/classes/csc373/zhanx223/assignment_4/output/image.png")
plt.close()

# Generate report text file
report_text = f"""
Radiology Image Dataset Report
------------------------------
1. Dataset Overview
   - Number of images: {num_images}
   - Image dimensions: {image_shape}
   - Color mode: {color_mode}

2. Pixel Intensity Statistics
   - Mean intensity: {pixel_mean:.2f}
   - Standard deviation: {pixel_std:.2f}
   - Minimum intensity: {pixel_min}
   - Maximum intensity: {pixel_max}

3. Label Distribution (pixel counts)
   - Label 0 (suspicious region): {label_counts[0]} pixels
   - Label 1 (suspicious region): {label_counts[1]} pixels
   - Label 2 (normal lung): {label_counts[2]} pixels
   - Label 3 (background): {label_counts[3]} pixels
"""

# Save report
with open("/deac/csc/classes/csc373/zhanx223/assignment_4/output/report.txt", "w") as f:
    f.write(report_text)

print("Report generated: /deac/csc/classes/csc373/zhanx223/assignment_4/output/report.txt")
