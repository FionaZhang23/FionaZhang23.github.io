import os
import numpy as np
predictions_path = "/deac/csc/classes/csc373/zhanx223/assignment_4/output/predictions.npy"
data_path = "/deac/csc/classes/csc373/data/assignment_4/labels_fall_2019.npy"  
predictions = np.load(predictions_path)
print(predictions.shape)
'''
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_multiple_label_maps(prediction_output, num_images_to_plot=100, save_path="/deac/csc/classes/csc373/zhanx223/assignment_4/scripts/predicted_label_maps.png"):
    """
    Plots and saves multiple predicted label maps from a one-hot encoded output.

    Parameters:
    prediction_output (numpy.ndarray): A 4D numpy array of shape (num_images, H, W, num_classes).
    num_images_to_plot (int): Number of images to plot (default: 100).
    save_path (str): File path to save the plot (default: "predicted_label_maps.png").
    """

    if prediction_output.ndim != 4:
        raise ValueError(f"Expected a 4D array, got shape {prediction_output.shape}")

    num_images, H, W, num_classes = prediction_output.shape
    num_images_to_plot = min(num_images, num_images_to_plot)  # Ensure we don't plot more than available

    # Determine the grid size for plotting
    grid_size = int(np.ceil(np.sqrt(num_images_to_plot)))  # Create a square layout

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i in range(num_images_to_plot):
        row, col = divmod(i, grid_size)
        label_map = np.argmax(prediction_output[i], axis=-1)  # Convert one-hot encoding to categorical labels

        ax = axes[row, col] if grid_size > 1 else axes[col]
        ax.imshow(label_map, cmap="gray")  # Plot in grayscale
        ax.axis("off")
        ax.set_title(f"Image {i+1}")

    # Hide unused subplots
    for i in range(num_images_to_plot, grid_size * grid_size):
        row, col = divmod(i, grid_size)
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    # Save plot to the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    save_full_path = os.path.join(script_dir, save_path)
    
    plt.savefig(save_full_path, dpi=300)  # Save with high resolution
    print(f"Plot saved at: {save_full_path}")

    plt.close()  # Close the plot to free memory

# Example usage
if __name__ == "__main__":
    # Load prediction output from a file (modify this path as needed)
    predictions_path = "/deac/csc/classes/csc373/zhanx223/assignment_4/output/predictions_sample.npy"  # Change this to your actual file path
    prediction_output = np.load(predictions_path)

    # Plot and save the label maps
    plot_multiple_label_maps(prediction_output, save_path="/deac/csc/classes/csc373/zhanx223/assignment_4/scripts/predicted_label_maps.png")
'''