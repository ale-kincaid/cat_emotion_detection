# Cat Emotion Detection

A computer vision project that detects whether an input image contains a cat, classifies the cat's emotional state, and estimates tail position as an additional visual cue. This project was developed for a Computer Vision course and combines deep learning with traditional image processing techniques.

## Project Overview

The goal of this project is to explore whether cat emotions can be predicted from image data using computer vision. The system uses a two-stage pipeline:

1. **Cat Detection**  
   A binary classifier determines whether the image contains a cat.

2. **Cat Emotion Classification**  
   If a cat is detected, a MobileNetV2-based model classifies the cat into one of several emotion categories.

3. **Tail Position Estimation**  
   OpenCV edge detection and contour analysis are used to estimate the cat's tail position. This tail estimate is then used as an additional heuristic to support the emotion prediction.

This project demonstrates image preprocessing, transfer learning, CNN-based classification, dataset splitting, model evaluation, and OpenCV-based feature extraction.

## Emotion Classes

The emotion classifier predicts one of the following classes:

```text
Angry
Disgusted
Happy
Normal
Sad
Scared
Surprised
