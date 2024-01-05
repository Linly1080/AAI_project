import os
import shutil
import random

# Define the dataset paths
base_path = '/data/linhuiyan/BIBM2023/AAI_project/processed_data'
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')
new_data_path = '//data/linhuiyan/BIBM2023/AAI_project/new_data'

# Create new data directories if they don't exist
new_train_path = os.path.join(new_data_path, 'train')
new_val_path = os.path.join(new_data_path, 'val')
os.makedirs(new_train_path, exist_ok=True)
os.makedirs(new_val_path, exist_ok=True)

# Function to get all file paths in a directory
def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Get all file paths from the original train and validation sets
original_train_files = get_file_paths(train_path)
original_val_files = get_file_paths(val_path)

# Calculate the number of files to move from train to validation to achieve an 8:2 split
total_files = len(original_train_files) + len(original_val_files)
desired_val_size = int(total_files * 0.2)
files_to_move = desired_val_size - len(original_val_files)

# Randomly select files from the original train set to move to the new validation set
selected_files_to_move = random.sample(original_train_files, files_to_move)

# Function to move files to a new directory
def move_files(files, new_path):
    for file in files:
        relative_path = os.path.relpath(file, base_path) # train/2/54187.npy
        path_parts = relative_path.split(os.path.sep)
        relative_path = os.path.join(*path_parts[1:])
        destination = os.path.join(new_path, relative_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        # print(destination)
        if len(destination.split(os.path.sep))!=8:
            print(False)
        shutil.copy(file, destination)

# Move the selected files to the new validation directory and the rest to the new training directory
move_files(selected_files_to_move, new_val_path)
move_files([f for f in original_train_files if f not in selected_files_to_move], new_train_path)

# Move all original validation files to the new validation directory without renaming
move_files(original_val_files, new_val_path)

