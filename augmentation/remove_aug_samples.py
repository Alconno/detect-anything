import os

image_dir = "./my_dataset/images/train"
label_dir = "./my_dataset/labels/train"
mask_dir = "./my_dataset/masks/train"

def remove_aug_files(dir_path):
    """
    Deletes all files in the specified directory whose filenames contain 'aug'.

    Parameters:
    ----------
    dir_path : str
        Path to the directory to scan for augmented files.
    """
    for filename in os.listdir(dir_path):
        if 'aug' in filename:
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

def remove_all_aug():
    remove_aug_files(image_dir)
    remove_aug_files(label_dir)
    remove_aug_files(mask_dir)

from multiprocessing import freeze_support

def main():
    freeze_support()
    
    remove_all_aug()

if __name__ == '__main__':
    main()
