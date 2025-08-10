import os
import shutil
import random
from collections import defaultdict

# paths
BASE_DIR = 'emotion_ds'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# create test directory if doesn't exist
os.makedirs(TEST_DIR, exist_ok=True)

# for each class, combine train+valid, shuffle, split 70/15/15, and move files
for class_name in os.listdir(TRAIN_DIR):
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    valid_class_dir = os.path.join(VALID_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)
    os.makedirs(test_class_dir, exist_ok=True)

    # gather all images from train and valid
    all_images = []
    for src_dir in [train_class_dir, valid_class_dir]:
        if os.path.exists(src_dir):
            all_images.extend([os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])

    random.shuffle(all_images)
    n_total = len(all_images)
    n_train = int(0.7 * n_total)
    n_valid = int(0.15 * n_total)
    n_test = n_total - n_train - n_valid

    train_imgs = all_images[:n_train]
    valid_imgs = all_images[n_train:n_train+n_valid]
    test_imgs = all_images[n_train+n_valid:]

    # move files to new folders
    def move_imgs(img_list, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for img_path in img_list:
            fname = os.path.basename(img_path)
            new_path = os.path.join(target_dir, fname)
            shutil.move(img_path, new_path)

    move_imgs(train_imgs, train_class_dir)
    move_imgs(valid_imgs, valid_class_dir)
    move_imgs(test_imgs, test_class_dir)

print('Dataset split complete. 70% train, 15% valid, 15% test.')
