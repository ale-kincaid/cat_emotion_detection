import tensorflow as tf
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import numpy as np
if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

gpus = tf.config.list_physical_devices('GPU') if hasattr(tf.config, 'list_physical_devices') else []
if gpus:
    print(f"✅ GPU detected: {gpus}")
else:
    print("⚠️ No GPU detected, running on CPU")

IMG_SIZE = (160, 160)
BATCH_SIZE = 64
AUTOTUNE = 4  

def parse_split_file(split_path):
    image_paths, labels = [], []
    with open(split_path, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            image_paths.append(path)
            labels.append(int(label))
    return image_paths, labels

def load_and_preprocess_image(path, label, augment=False):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        # normalize [0,1]
        image = image / 255.0  
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label
    except Exception:
        # Return dummy image and label if error occurs
        return tf.zeros(IMG_SIZE + (3,)), label


# creates tf datset 
def build_dataset(split_path, augment=False):
    image_paths, labels = parse_split_file(split_path)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if augment:
        ds = ds.map(lambda x, y: load_and_preprocess_image(x, y, True), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000).batch(BATCH_SIZE)
    try:
        ds = ds.prefetch(AUTOTUNE)
    except Exception:
        pass
    return ds

train_ds = build_dataset('cat_train.txt', augment=True)
val_ds = build_dataset('cat_val.txt')
test_ds = build_dataset('cat_test.txt')

l2_reg = regularizers.l2(0.001)

# Load MobileNetV2 base model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  

inputs = Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2_reg)(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[callback])
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy: {:.2f}".format(test_acc))

# Save model in HDF5 format
model.save('cat_detector_model.h5')