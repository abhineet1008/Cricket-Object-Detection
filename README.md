# Cricket-Object-Detection

This project focuses on object detection in cricket images, specifically identifying bats, balls, stumps, or no object within an image using handcrafted feature extraction methods.

## Project Overview

Automatically identify cricket-related objects in static images and videos by using hand-crafted engineering features for identifying:
* Cricket Bat
* Cricket Ball
* Cricket Stumps
* No Object

## Given Problem

Classify regions of a cricket image into classes of background `[0]`, ball `[1]`, bat `[2]`, and stumps `[3]` following certain rules of assignment:
* Divide the image into regions as cells.
* Each cell shall be given a label for background, ball, bat, or stumps.
* If there is more than one object or part of an object in a cell, the label is assigned with the following priority: ball > bat > stumps > background.
* Images should be resized to 800 × 600 pixels. Each image should be divided into an 8x8 grid, resulting in 64 cells.
* The classification will be done at the cell level.

## Technologies Used

* **Language:** Python
* **Libraries:** NumPy
* **Classical ML:** Evaluated KNN, SVM, Random Forest, and XGBoost with carefully engineered features.
## Feature Engineering
* **Combined color (HSV histograms & stats), texture (LBP), and shape/edge (HOG) features.**

## Feature Engineering

- *Combined color (HSV histograms & stats), texture (LBP), and shape/edge (HOG) features.*

| Object | Challenge Scenario | HOG Fails (individually) Because | HSV Fails (individually) Because | LBP Fails (individually) Because |
| :--- | :--- | :--- | :--- | :--- |
| **Bat** | Horizontal/Angular Swings | Gradient orientation flips 90°. | Works okay, but area is small. | Motion blur destroys pattern. |
| **Ball** | Crowd Background | Too many "round" shapes in crowd. | Red shirts in crowd mimic ball. | Crowd texture mimics ball noise. |
| **Stumps** | Occlusion | "3-poles" shape is blocked/merged. | Works well if any wood is visible. | Leg pads have similar texture. |

## Challenges

* **Ball:** Detection is difficult due to extreme size variation, motion blur, and color confusion with background elements.
* **Bat:** Detection is challenging because of its varying orientation, elongated shape, and frequent occlusion by the player.
* **Stumps:** Detection is hard due to partial visibility, occlusion, and visual similarity to other vertical objects in the scene.
* **Data:** Very imbalanced, as cells with objects are far fewer compared to background cells. The uneven distribution makes it harder for models to learn the minority classes.

## Results

**Overall Performance**
* High overall accuracy (89%). Accuracy is largely influenced by the Background class, which makes up ~87.5% of the dataset (3663 out of 4186 samples).
* A trained and validated classification model capable of cell-wise prediction.
* A structured CSV output containing predicted class labels for all grid cells in each image.

**Class-Specific Performance**
* **Background (Excellent):** Precision: 0.96 | Recall: 0.93 | F1-score: 0.94
  * _The model reliably identifies background regions._
* **Ball (Moderate):** Precision: 0.65 | Recall: 0.66 | F1-score: 0.66
  * _The model detects balls reasonably well but still misses ~34% of instances._
* **Stumps (Best Object Class):** Precision: 0.60 | Recall: 0.73 | F1-score: 0.66
  * _Stumps are detected more consistently than other objects._
* **Bat (Weakest Class):** Precision: 0.32 | Recall: 0.38 | F1-score: 0.35
  * _A large number of bat regions are misclassified as background or confused with other objects._

**Recall Imbalance**
* The model struggles to consistently detect smaller or visually variable objects, especially bats (Ball recall: 0.66 | Bat recall: 0.38 | Stump recall: 0.73).

## Conclusion

* A machine learning-based approach was successfully developed to detect balls, bats, and stumps from cricket images using hand-crafted visual features.
* Combining shape (HOG), color (HSV statistics and histograms), and texture (LBP) features enabled effective representation of image regions.
* XGBoost outperformed other models under class imbalance.
* Building a reliable ML system requires correct data handling, thoughtful evaluation, and end-to-end consistency.
