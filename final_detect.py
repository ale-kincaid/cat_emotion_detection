import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
from cat_emotion_detector import estimate_tail_position, combine_emotion_and_tail, EMOTION_CLASSES

IMG_SIZE = (160, 160)

# load models
cat_model = load_model('cat_detector_model.h5')
emotion_model = load_model('cat_emotion_mobilenetv2.h5')

def preprocess_img(img_path):
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def detect_and_visualize(img_path):
    img_input = preprocess_img(img_path)
    cat_prob = cat_model.predict(img_input)[0][0]
    img_bgr = cv2.imread(img_path)

    if cat_prob < 0.5:
        cv2.putText(img_bgr, "No cat detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # resize for display (e.g., width=640)
        display_width = 640
        h, w = img_bgr.shape[:2]
        if w > display_width:
            scale = display_width / w
            img_bgr = cv2.resize(img_bgr, (display_width, int(h * scale)))
        cv2.imshow("Result", img_bgr)
        cv2.waitKey(0)
        return

    # if cat detected, run emotion detector
    emotion_probs = emotion_model.predict(img_input)[0]
    tail_pos, tail_conf = estimate_tail_position(img_path)
    emotion_label, combined_conf = combine_emotion_and_tail(emotion_probs, tail_pos, tail_conf)

    # visualize
    label_text = f"{emotion_label} ({combined_conf:.2f})"
    # larger, bolder text for emotion/confidence
    cv2.putText(img_bgr, label_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 8.0, (0,255,0), 16)
    # larger, bolder text for tail position
    cv2.putText(img_bgr, f"Tail: {tail_pos}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255,255,0), 12)
    
    # resize for display (e.g., width=640)
    display_width = 640
    h, w = img_bgr.shape[:2]
    if w > display_width:
        scale = display_width / w
        img_bgr = cv2.resize(img_bgr, (display_width, int(h * scale)))
    cv2.imshow("Result", img_bgr)
    cv2.waitKey(0)

if __name__ == "__main__":
    import sys
    img_path = "small_test_images/IMG_6871.jpg" 
    detect_and_visualize(img_path)