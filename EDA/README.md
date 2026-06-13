# Dataset Analysis and Feature Engineering

## Overview
To properly understand our dataset and prepare for model training, we conducted a systematic analysis across three key dimensions. Each step reveals critical information about data quality and potential training challenges. The three dimensions are Data Imbalance Assessment, Outlier Detection, and Spatial Distribution Analysis.

---

## 1. Data Distribution & Imbalance
An evaluation of class distribution across the dataset was performed to identify potential bias towards specific object categories or background pixels. The dataset exhibits an extreme class imbalance. 

* **Approximately 90%** of the training data consists of background pixels.
* **The breakdown of total grid cells is:** Background at 18,430 (87.5%), Stumps at 1,200 (5.7%), Bats at 717 (3.4%), and Balls at 709 (3.4%).
* **Training Risk:** Due to this imbalance, the model risks learning to predict "Background" for all inputs to achieve high accuracy, while completely failing to detect actual objects.

### Outlier Detection
We utilized box plots to identify anomalous image characteristics that could negatively impact model generalization during training. 
* The box plots show that the number of grid cells occupied by objects generally follows the trend: **Balls < Bat < Stump**.
* Outliers are present and may require data cleaning if they bias the model.

---

## 2. Spatial Distribution Analysis
We mapped the frequency and location patterns of the objects (bat, ball, stumps) to understand positional biases within the training images.

* **Dynamic Object (The Ball):** Has the widest spread of the three objects, fading out gradually toward the edges. This is expected, as the ball is the only object that freely moves through the air across the entire frame.
* **Action Object (The Bat):** Corresponds to the typical batting stance where the batsman stands upright. The bat rarely touches the bottom rows or floats at the very top edge of the frame.
* **Anchored Object (The Stump):** Is heavily concentrated in a block at the bottom-center of the grid. While stumps are stationary objects fixed into the ground, they are captured from different angles, which causes them to sometimes appear at slightly different locations.

---

## 3. Dimensionality Reduction & Clustering
To assess the separability of our handcrafted features, we applied dimensionality reduction techniques on 10,000 samples.

* **PCA (Principal Component Analysis):** Revealed a blob-like structure with no clear separation between the different object classes. This demonstrates that linear dimensionality reduction fails to capture the complex, non-linear relationships in the image data, indicating that linear regression would not work.
* **t-SNE:** Similarly showed no distinct clustering patterns, apart from minimal clustering for stumps. Because of this, it is expected that non-linear models such as XGBoost or Random Forest will make better guesses.

---

## 4. Feature Engineering
A multi-feature representation was utilized to describe each image cell, successfully capturing shape, color, and texture information:

* **HOG (Histogram of Oriented Gradients):** Used for structural and edge-based features to capture the structural edges of bats, balls, and stumps.
* **Color Histograms:** Implemented 16-bin histograms for Hue, Saturation, and Value for distinguishing object colors, to help identify specific colors of the ball (Red) or field (Green).
* **HSV Color Statistics:** The mean and standard deviation of HSV values were calculated for color consistency to provide a global color context for each image patch.
* **LBP (Local Binary Patterns):** Implemented for texture discrimination, allowing the model to distinguish between different surfaces, such as the smooth surface of the stumps versus the rough texture of the grass.
