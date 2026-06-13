# %% [markdown]
# # 🏏 Cricket Object Detection (Non-Deep Learning Approach)
# 
# **Project Goal:** Detect Bat, Ball, and Stumps in cricket images using Hand-Crafted Features and XGBoost.
# 
# **Workflow:**
# 1.  **Feature Extraction:** Extract Shape (HOG), Texture (LBP), and Color information from image patches.
# 2.  **Model Training:** Train an XGBoost classifier with regularization to handle overfitting.
# 3.  **Inference:** Predict on new images using a sliding window approach with smart thresholding.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from skimage.color import rgb2gray

# Feature Extraction Libraries
from skimage.feature import hog, local_binary_pattern

# Machine Learning Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# %%
# --- GLOBAL CONFIGURATION ---

# Paths
DATA_FOLDER = r"C:\Dataset_new\dataset_processed"   # Folder containing your 800x600 images
CSV_FILE = r"C:\Dataset_new\annotations.csv"             # Your labels file
MODEL_FILENAME = "model_final_xgb.pkl"

# Image & Grid Dimensions (Must match your Tagging Tool)
IMG_WIDTH = 800
IMG_HEIGHT = 600
ROWS = 8
COLS = 8
CELL_W = IMG_WIDTH // COLS  # 100 pixels
CELL_H = IMG_HEIGHT // ROWS # 75 pixels

# Class Definitions
LABELS = {0: "Background", 1: "Ball", 2: "Bat", 3: "Stump"}


print(" IMAGES_DIR:", DATA_FOLDER)
print(" ANNOTATIONS_CSV:", CSV_FILE)


# %%
def safe_read_csv(path):
    encodings = ['utf-8', 'latin1', 'cp1252']
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_exc = e
    raise last_exc

ann = safe_read_csv(CSV_FILE)
if 'ImageFileName' not in ann.columns:
    raise ValueError("annotation CSV must have an 'ImageFileName' column.")
print("Loaded annotations:", ann.shape)
ann.head()

# %% [markdown]
# ## 2. Feature Extraction Functions
# We use a combination of features to describe every $100 \times 75$ cell:
# 1.  **HOG:** Captures Shapes (Edges of Bat/Stumps).
# 2.  **Color Histogram:** Captures Color Distribution (Red Ball vs Green Grass).
# 3.  **Color Stats:** Mean and Std Dev of HSV values.
# 4.  **LBP:** Captures Texture (Smooth Bat vs Rough Grass).

# %%
def get_hog_features(gray_img):
    """Extracts Histogram of Oriented Gradients (Shape)"""
    try:
        # settings: 9 orientations, 16x16 pixels per cell, 2x2 cells per block
        features = hog(gray_img, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        return features
    except:
        # Fallback for edge cases
        return np.zeros(1)

def get_color_histogram(hsv_img, bins=16):
    """Extracts Color Histograms for Hue, Saturation, Value"""
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    
    # Normalize to make it scale-invariant
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    return np.concatenate([hist_h, hist_s, hist_v]).flatten()

def get_color_stats(hsv_img):
    """Extracts simple statistical features (Mean/Std Dev) from color channels"""
    mean, std = cv2.meanStdDev(hsv_img)
    return np.concatenate([mean, std]).flatten()

def get_lbp_features(gray_img, P=8, R=1):
    """Extracts Local Binary Patterns (Texture)"""
    lbp = local_binary_pattern(gray_img, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    
    # Normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# %% [markdown]
# ## 3. Data Processing Pipeline
# Reads images, slices them into grid cells, generates features, and saves them as Numpy arrays.

# %%
def prepare_dataset():
    # Check if data is already processed to save time
    if os.path.exists('X_final.npy') and os.path.exists('y_final.npy'):
        print("✅ Found existing processed data. Loading from .npy files...")
        X = np.load('X_final.npy')
        y = np.load('y_final.npy')
        return X, y

    # Load CSV
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found. Ensure working directory is correct.")
        return None, None
        
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    print(f"📂 Processing {len(df)} images from {DATA_FOLDER}...")

    X_list = []
    y_list = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row['ImageFileName']
        path = os.path.join(DATA_FOLDER, filename)
        
        # Load and validate image
        img = cv2.imread(path)
        if img is None: 
            continue
        
        # Resize to standard dimensions
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Pre-convert to Gray and HSV to avoid doing it per-cell (optimization)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Grid Slicing Loop
        cell_counter = 0
        for r in range(ROWS):
            for c in range(COLS):
                cell_counter += 1
                
                # Calculate Coordinates
                y1, y2 = r * CELL_H, (r + 1) * CELL_H
                x1, x2 = c * CELL_W, (c + 1) * CELL_W
                
                # Crop Patches
                patch_gray = img_gray[y1:y2, x1:x2]
                patch_hsv = img_hsv[y1:y2, x1:x2]
                
                # Feature Extraction
                f_hog = get_hog_features(patch_gray)
                f_hist = get_color_histogram(patch_hsv)
                f_stats = get_color_stats(patch_hsv)
                f_lbp = get_lbp_features(patch_gray)
                
                # Concatenate all features into one vector
                features = np.concatenate([f_hog, f_hist, f_stats, f_lbp])
                
                # Get Label from CSV
                col_name = f"c{cell_counter:02d}"
                label = int(row[col_name])
                
                X_list.append(features)
                y_list.append(label)

    # Convert lists to Numpy Arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Save for future use
    np.save('X_final.npy', X)
    np.save('y_final.npy', y)
    print("✅ Data processing complete and saved to .npy files.")
    return X, y

# Run Data Prep
X, y = prepare_dataset()
print(X)

# %% [markdown]
# ## 4. Model Training (XGBoost)
# We use **Sample Weights** to handle the severe class imbalance (95% background, 5% objects).

# %%
# 1. Train/Test Split
print("✂️ Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y  # Essential to ensure Test set has Balls/Bats/Stumps
)

# 2. Calculate Sample Weights
# This tells XGBoost: "Pay 20x more attention to a Ball than to Background"
print("⚖️ Computing Sample Weights...")
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)


# 3. Model Definition (Config provided by User)
print("🚀 Initializing XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=600,      # Number of trees
    max_depth=5,           # Depth of trees (Reduced to prevent overfitting)
    learning_rate=0.05,    # Step size
    objective='multi:softprob', # Multiclass probability output
    num_class=4,           # 4 Classes (0,1,2,3)
    eval_metric='mlogloss',
    tree_method='hist',    # Faster training on large data
    reg_alpha=0.1,         # L1 Regularization (Lasso)
    reg_lambda=10.0,      # L2 Regularization (Default)
    subsample=1.0,         # Use 80% of data per tree
    colsample_bytree=1.0,  # Use 80% of features per tree
    random_state=42,
    min_child_weight= 5, 
    gamma=1.0
    

)

# 4. Training
print("🏋️ Training Model (This may take a minute)...")
model.fit(
    X_train, 
    y_train, 
    sample_weight=sample_weights,
    verbose=False
)
print("✅ Training Complete.")



# %%

# 3. Model Definition (Config provided by User)
print("🚀 Initializing XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=600,      # Number of trees
    max_depth=5,           # Depth of trees (Reduced to prevent overfitting)
    learning_rate=0.05,    # Step size
    objective='multi:softprob', # Multiclass probability output
    num_class=4,           # 4 Classes (0,1,2,3)
    eval_metric='mlogloss',
    tree_method='hist',    # Faster training on large data
    reg_alpha=0.1,         # L1 Regularization (Lasso)
    reg_lambda=10.0,      # L2 Regularization (Default)
    subsample=1.0,         # Use 80% of data per tree
    colsample_bytree=1.0,  # Use 80% of features per tree
    random_state=42,
    min_child_weight= 5, 
    gamma=1.0
    

)

# 4. Training
print("🏋️ Training Model (This may take a minute)...")
model.fit(
    X_train, 
    y_train, 
    sample_weight=sample_weights,
    verbose=False
)
print("✅ Training Complete.")

# Predict on Test Set
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Overall Test Accuracy: {acc:.2%}")


print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Background', 'Ball', 'Bat', 'Stump']))

# %% [markdown]
# K-Nearest Neighbors (KNN)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Training KNN...")

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),   # REQUIRED for KNN
    ("knn", KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",         # helps with class imbalance
        metric="minkowski"
    ))
])

knn_pipeline.fit(X_train, y_train)

y_pred_knn = knn_pipeline.predict(X_test)

print("\n📊 KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\n--- KNN Classification Report ---")
print(classification_report(
    y_test, y_pred_knn,
    target_names=['Background', 'Ball', 'Bat', 'Stump']
))


# %% [markdown]
# Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",   # VERY important for your data
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n📊 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\n--- Random Forest Classification Report ---")
print(classification_report(
    y_test, y_pred_rf,
    target_names=['Background', 'Ball', 'Bat', 'Stump']
))


# %% [markdown]
# Support Vector Classifier (SVC)

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Training SVC...")

svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),   # REQUIRED
    ("svc", SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        class_weight="balanced",    # CRITICAL for your dataset
        probability=False
    ))
])

svc_pipeline.fit(X_train, y_train)

y_pred_svc = svc_pipeline.predict(X_test)

print("\n📊 SVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print("\n--- SVC Classification Report ---")
print(classification_report(
    y_test, y_pred_svc,
    target_names=['Background', 'Ball', 'Bat', 'Stump']
))


# %% [markdown]
# ## 5. Compare all the Model's classification report

# %%
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# %%
def evaluate_model(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

# %%
results = []

# XGBoost
y_pred_xgb = model.predict(X_test)
results.append(evaluate_model("XGBoost", y_test, y_pred_xgb))

# KNN
y_pred_knn = knn_pipeline.predict(X_test)
results.append(evaluate_model("KNN", y_test, y_pred_knn))

# Random Forest
y_pred_rf = rf_model.predict(X_test)
results.append(evaluate_model("Random Forest", y_test, y_pred_rf))

# SVC
y_pred_svc = svc_pipeline.predict(X_test)
results.append(evaluate_model("SVC (RBF)", y_test, y_pred_svc))

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values(by="F1-score (Macro)", ascending=False)
pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

print("\n📊 MODEL COMPARISON SUMMARY\n")
print(comparison_df.round(4))


# %%



