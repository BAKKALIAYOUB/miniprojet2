from CalculeCaracteristics import caracGlobale
import pandas as pd
import os
from PIL import Image
import numpy as np


def process_images(base_folder, output_csv):
    data = []

    # Walk through the directories and subdirectories in base_folder
    for category in os.listdir(base_folder):
        category_path = os.path.join(base_folder, category)

        if os.path.isdir(category_path):  # Ensure we're looking at folders
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.obj')):  # You can add other image formats if needed
                    model_path = os.path.join(category_path, filename)

                    # Get the global features
                    features = caracGlobale(mesh_path=model_path, reduce_factor=0.7)
                    mount_image_path = f"/{category}/{filename}"
                    # Append the data in the required format
                    data.append([mount_image_path, category] + features.flatten().tolist())

    # Create a DataFrame and save it to a CSV
    df = pd.DataFrame(data, columns=['modelpath', 'Category'] + [f'Feature_{i + 1}' for i in range(len(data[0]) - 2)])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved at {output_csv}")


if __name__ == "__main__":
    # Call the function with your folder and desired output file name
    base_folder = "./3D Models/"
    output_csv = "features_70.csv"
    process_images(base_folder, output_csv)
