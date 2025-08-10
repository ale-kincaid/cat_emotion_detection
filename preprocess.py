import os
import random

# paths
ANNOTATIONS_PATH = 'annotations/list.txt'
IMAGES_DIR = 'images'
TRAIN_FILE = 'cat_train.txt'
VAL_FILE = 'cat_val.txt'
TEST_FILE = 'cat_test.txt'

# split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def main():
    # read annotation file, skip first 6 header lines
    with open(ANNOTATIONS_PATH, 'r') as f:
        lines = f.readlines()[6:]

    all_images = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        image_name = parts[0] + '.jpg'
        species = int(parts[2])  # 1 for cat, 2 for dog
        label = 1 if species == 1 else 0
        image_path = os.path.join(IMAGES_DIR, image_name)
        all_images.append(f"{image_path} {label}")

    # shuffle list to randomize
    random.shuffle(all_images)

    # split
    n_total = len(all_images)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_test = n_total - n_train - n_val

    train_split = all_images[:n_train]
    val_split = all_images[n_train:n_train + n_val]
    test_split = all_images[n_train + n_val:]

    # save splits
    with open(TRAIN_FILE, 'w') as f:
        f.write('\n'.join(train_split))
    with open(VAL_FILE, 'w') as f:
        f.write('\n'.join(val_split))
    with open(TEST_FILE, 'w') as f:
        f.write('\n'.join(test_split))

    print(f"Saved {len(train_split)} train, {len(val_split)} val, {len(test_split)} test samples.")

if __name__ == "__main__":
    main()