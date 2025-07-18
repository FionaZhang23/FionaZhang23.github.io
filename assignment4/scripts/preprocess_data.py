import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, decode_labels, reshape_data
from sklearn.preprocessing import OneHotEncoder

data_path = "/deac/csc/classes/csc373/data/assignment_4"
output_path = "/deac/csc/classes/csc373/zhanx223/assignment_4/output"

scan_fall_2019 = os.path.join(data_path, "scan_fall_2019.npy")
labels_fall_2019 = os.path.join(data_path, "labels_fall_2019.npy")

print("Loading dataset...")
scans = np.load(scan_fall_2019)
labels = np.load(labels_fall_2019)

left_crop = 180
right_crop = 100
top_crop = 20
bottom_crop = 20  

num_samples, h, w, _ = scans.shape
x1, x2 = left_crop, w - right_crop
y1, y2 = top_crop, h - bottom_crop

if x1 >= x2 or y1 >= y2:
    raise ValueError("Invalid crop dimensions! Check your cropping parameters.")

print(f"Cropping images and labels: left={left_crop}, right={right_crop}, top={top_crop}, bottom={bottom_crop}...")
cropped_scans = scans[:, y1:y2, x1:x2]
cropped_labels = labels[:, y1:y2, x1:x2]

labels_decoded = decode_labels(cropped_labels)
threshold = 0.95

non_background_indices = [
    i for i in range(len(labels_decoded)) 
    if np.mean((labels_decoded[i] == 3).astype(np.float32)) < threshold
]
filtered_scans = cropped_scans[non_background_indices]
filtered_labels = labels_decoded[non_background_indices]

print(f"Removed {len(cropped_scans) - len(filtered_scans)}.")
print(f"Final dataset shape: {filtered_scans.shape}")

filtered_scans_path = os.path.join(output_path, "filtered_scans.npy")
filtered_labels_path = os.path.join(output_path, "filtered_labels.npy")

np.save(filtered_scans_path, filtered_scans)
np.save(filtered_labels_path, filtered_labels)

print(f"Filtered dataset saved at: {filtered_scans_path}")
print(f"Filtered labels saved at: {filtered_labels_path}")

num_samples_to_plot = 5
random_indices = np.random.choice(cropped_scans.shape[0], num_samples_to_plot, replace=False)

fig, axes = plt.subplots(1, num_samples_to_plot, figsize=(num_samples_to_plot * 3, 3))

for i, idx in enumerate(random_indices):
    cropped_image = cropped_scans[idx].astype(np.float32)

    axes[i].imshow(cropped_image, cmap='gray')
    axes[i].set_title(f"Cropped {idx}")
    axes[i].axis("off")

plt.tight_layout()

plot_path = os.path.join(output_path, "cropped_images.png")
plt.savefig(plot_path, dpi=300)
plt.close()
