import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Directories
output_dir = '/deac/csc/classes/csc373/zhanx223/assignment_3/output'
train_data_dir = '/deac/csc/classes/csc373/data/assignment_3/train'
report_path = os.path.join(output_dir, 'data_quality_report.txt')

def check_image_quality(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Check if image is valid
        img = Image.open(image_path).convert("RGB")  # Reload in case verify closes the file
        width, height = img.size
        aspect_ratio = round(width / height, 2)
        mean_color = np.array(img).mean(axis=(0, 1))  # Mean RGB values
        return width, height, aspect_ratio, mean_color.tolist()
    except Exception as e:
        return None, None, None, str(e)

def detect_duplicates(report_df):
    duplicates = report_df['Filename'].duplicated().sum()
    return duplicates

def detect_extreme_aspect_ratios(report_df, threshold=3.0):
    extreme_aspect_count = ((report_df['Aspect Ratio'] > threshold) | (report_df['Aspect Ratio'] < 1/threshold)).sum()
    return extreme_aspect_count

def generate_report(image_dir):
    report_data = []
    
    for root, _, files in os.walk(image_dir):
        for file in tqdm(files):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                width, height, aspect_ratio, quality_info = check_image_quality(image_path)
                report_data.append([file, width, height, aspect_ratio, quality_info])
    
    df = pd.DataFrame(report_data, columns=['Filename', 'Width', 'Height', 'Aspect Ratio', 'Mean RGB / Error'])
    
    # Summary statistics
    valid_images = df.dropna(subset=['Width'])
    num_images = len(df)
    num_valid = len(valid_images)
    num_corrupt = num_images - num_valid
    avg_width = valid_images['Width'].mean()
    avg_height = valid_images['Height'].mean()
    avg_aspect_ratio = valid_images['Aspect Ratio'].mean()
    num_duplicates = detect_duplicates(df)
    num_extreme_aspect = detect_extreme_aspect_ratios(valid_images)
    
    summary = f"""
    Data Quality Report for {image_dir}
    -----------------------------------
    Total images: {num_images}
    Valid images: {num_valid}
    Corrupt images: {num_corrupt}
    Duplicate images: {num_duplicates}
    Images with extreme aspect ratios: {num_extreme_aspect}
    
    Image Statistics (Valid Images Only):
    Average Width: {avg_width:.2f}
    Average Height: {avg_height:.2f}
    Average Aspect Ratio: {avg_aspect_ratio:.2f}
    """
    
    with open(report_path, 'w') as f:
        f.write(summary)
    
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    generate_report(train_data_dir)
