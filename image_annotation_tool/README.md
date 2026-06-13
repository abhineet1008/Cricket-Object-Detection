# Image Annotation Tool 🏏

This directory contains the Image Annotation Tool used to create custom datasets for the Cricket Object Detection project. It allows users to manually label cricket-specific objects in frames/images to train object detection models (such as YOLO, Faster R-CNN, etc.).

## 🎯 Purpose
Accurate bounding box annotations are crucial for training robust computer vision models. This tool streamlines the process of drawing bounding boxes around key cricket elements, specifically:
* **Bat**
* **Ball**
* **Stumps**

## ✨ Features
* **Intuitive Interface:** Easily load images from a directory and draw bounding boxes.
* **Multi-Class Support:** Tag objects as `bat`, `ball`, or `stumps`.
* **Standardized Output:** Saves annotations in standard formats compatible with modern object detection frameworks.

## 🛠️ Prerequisites
Ensure you have Python installed along with the necessary dependencies. You can install the required packages using:

```bash
pip install -r requirements.txt
