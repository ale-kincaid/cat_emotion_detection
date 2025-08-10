from PIL import Image
import os

def remove_corrupt_images(image_dir):
    num_removed = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()  # Will not load the image, but will check for corruption
                except Exception:
                    print(f"Removing corrupt image: {path}")
                    os.remove(path)
                    num_removed += 1
    print(f"Removed {num_removed} corrupt images.")



remove_corrupt_images('images')