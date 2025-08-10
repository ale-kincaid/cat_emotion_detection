import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EMOTION_CLASSES = ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised']
NUM_CLASSES = len(EMOTION_CLASSES)

# try to estimate tail position with canny and contours
def estimate_tail_position(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        return 'Unknown', 0.0

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # second largest contour hopefully tail 
    tail_contour = contours[1]

    # compute Hu moments
    hu_moments = cv2.HuMoments(cv2.moments(tail_contour)).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # use simple thresholds (tuned heuristically)
    compactness = hu_moments[0]
    elongation = hu_moments[1]

    # example mapping (adjust thresholds based on real data if needed)
    if elongation < -2.5:
        tail_pos = 'Straight'
    elif compactness > 0 and elongation > -2.0:
        tail_pos = 'Curled'
    elif elongation > -3.0 and compactness < -0.5:
        tail_pos = 'Up'
    else:
        tail_pos = 'Down'

    # confidence from contour area
    confidence = cv2.contourArea(tail_contour) / (IMG_SIZE[0] * IMG_SIZE[1])
    return tail_pos, confidence

# combine tail score and emotion score
def combine_emotion_and_tail(emotion_probs, tail_pos, tail_conf):
    emotion_idx = np.argmax(emotion_probs)
    emotion_label = EMOTION_CLASSES[emotion_idx]
    emotion_conf = np.max(emotion_probs)
    # boost confidence if tail matches expected
    EMOTION_TAIL_MAP = {
        'Angry': ['Puffed'],
        'Disgusted': ['Down'],
        'Happy': ['Up', 'Straight'],
        'Normal': ['Straight'],
        'Sad': ['Down', 'Curled'],
        'Scared': ['Puffed', 'Curled'],
        'Surprised': ['Up', 'Puffed']
    }
    if tail_pos in EMOTION_TAIL_MAP.get(emotion_label, []):
        combined_conf = min(emotion_conf + 0.1 * tail_conf, 1.0)
    else:
        combined_conf = emotion_conf
    return emotion_label, combined_conf

# visualize confusion matrix
def visualize_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes, square=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

# training pipeline
if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    train_ds = train_datagen.flow_from_directory(
        'emotion_ds/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_ds = val_datagen.flow_from_directory(
        'emotion_ds/valid',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    test_ds = test_datagen.flow_from_directory(
        'emotion_ds/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    l2_reg = regularizers.l2(0.001)
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2_reg)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[callback])
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_acc:.2f}")
    
    # save model in HDF5 format 
    model.save('cat_emotion_mobilenetv2.h5')

    # test set evaluation
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.2f}")

    # confusion matrix and classification report
    print("\nGenerating Confusion Matrix...")
    
    # get predictions and true labels
    y_true = []
    y_pred = []
    emotion_only_preds = []
    combined_preds = []
    test_dir = 'Data/emotion_ds/test'  # Base directory for test images

    # process each image in the test set
    for i, (img_path, true_label) in enumerate(zip(test_ds.filenames, test_ds.classes)):
        if i % 100 == 0:  # show progress every 100 images
            print(f"Processing image {i+1}/{len(test_ds.filenames)}")
            
        # get full path to image
        full_img_path = os.path.join(test_dir, img_path)
        
        try:
            # load and preprocess image
            img = tf.keras.preprocessing.image.load_img(full_img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # get emotion prediction
            emotion_probs = model.predict(img_array, verbose=0)[0]
            emotion_pred = np.argmax(emotion_probs)
            
            # store emotion-only prediction
            emotion_only_preds.append(emotion_pred)
            
            # get tail position
            tail_pos, tail_conf = estimate_tail_position(full_img_path)
            
            # get combined prediction
            emotion_label, _ = combine_emotion_and_tail(emotion_probs, tail_pos, tail_conf)
            combined_preds.append(EMOTION_CLASSES.index(emotion_label))
            
            # store predictions and true label
            y_true.append(true_label)
            y_pred.append(emotion_pred)
        except Exception as e:
            print(f"Error processing image {full_img_path}: {str(e)}")
            continue
    
    # Generate confusion matrix and reports
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report (Emotion Only):")
    print(classification_report(y_true, y_pred, target_names=EMOTION_CLASSES))
    
    # visualize confusion matrix using seaborn
    visualize_confusion_matrix(cm, EMOTION_CLASSES, "Cat Emotion Detection Confusion Matrix")

    # generate combined predictions report
    print("\n=== Emotion Only ===")
    print(classification_report(y_true, emotion_only_preds, target_names=EMOTION_CLASSES))
    print("\n=== Emotion + Tail Position ===")
    print(classification_report(y_true, combined_preds, target_names=EMOTION_CLASSES))
