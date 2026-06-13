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
DATA_FOLDER = "dataset_processed"   # Folder containing your 800x600 images
CSV_FILE = "annotations.csv"        # Your labels file
MODEL_FILENAME = "group_08_pickle.pkl"

# Image & Grid Dimensions (Must match your Tagging Tool)
IMG_WIDTH = 800
IMG_HEIGHT = 600
ROWS = 8
COLS = 8
CELL_W = IMG_WIDTH // COLS  # 100 pixels
CELL_H = IMG_HEIGHT // ROWS # 75 pixels

# Class Definitions
LABELS = {0: "Background", 1: "Ball", 2: "Bat", 3: "Stump"}

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

# %% [markdown]
# ## 5. Evaluation & Analysis

# %%
# Predict on Test Set
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Overall Test Accuracy: {acc:.2%}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Background', 'Ball', 'Bat', 'Stump']))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Back', 'Ball', 'Bat', 'Stump'],
            yticklabels=['Back', 'Ball', 'Bat', 'Stump'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance Plot
# This graph is excellent for explaining WHY the model works to stakeholders
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15, height=0.5, importance_type='weight', title='Top 15 Features')
plt.show()

# Save Model
with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump(model, f)
print(f"💾 Model saved as {MODEL_FILENAME}")

# %% [markdown]
# ## 6. Inference: Prediction on New Images
# This function applies the model to a new image. It includes **Smart Thresholding** to rescue the "Bat" class, which is often harder to detect than the Ball or Stumps.

# %%
def predict_and_visualize(image_path, model):
    # Load Image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return

    # Resize
    img = cv2.resize(original_img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Pre-process for features
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    features_list = []
    coordinates = []

    # 1. Feature Extraction Loop
    for r in range(ROWS):
        for c in range(COLS):
            y1, y2 = r * CELL_H, (r + 1) * CELL_H
            x1, x2 = c * CELL_W, (c + 1) * CELL_W
            
            patch_gray = img_gray[y1:y2, x1:x2]
            patch_hsv = img_hsv[y1:y2, x1:x2]
            
            f_hog = get_hog_features(patch_gray)
            f_hist = get_color_histogram(patch_hsv)
            f_stats = get_color_stats(patch_hsv)
            f_lbp = get_lbp_features(patch_gray)
            
            combined = np.concatenate([f_hog, f_hist, f_stats, f_lbp])
            features_list.append(combined)
            coordinates.append((x1, y1, x2, y2))

    # 2. Batch Inference
    X_input = np.array(features_list)
    probs = model.predict_proba(X_input) # Get probabilities for all classes

    # Visualization Setup
    overlay = img.copy()
    colors = {2: (0, 0, 255), 3: (255, 0, 0), 1: (0, 255, 0)} # Red, Blue, Green (BGR)
    
    # --- SMART THRESHOLDS ---
    # Adjust these based on validation results
    THRESHOLDS = {
        1: 0.60,  # Ball: High confidence required (reduce false positives)
        2: 0.30,  # Bat: Lower confidence allowed (hard to detect)
        3: 0.50   # Stump: Medium confidence
    }

    # 3. Decision Logic
    for i, prob_vector in enumerate(probs):
        # prob_vector = [Prob_Back, Prob_Ball, Prob_Bat, Prob_Stump]
        pred_class = np.argmax(prob_vector)
        score = prob_vector[pred_class]
        
        # -- OVERRIDE LOGIC --
        # If model thinks it's Background, but Bat probability is decent (>0.30), 
        # force it to be Bat. This helps recall.
        if pred_class == 0:
            if prob_vector[2] > THRESHOLDS[2]:
                pred_class = 2
                score = prob_vector[2]

        # Final Threshold Check
        if pred_class != 0 and score >= THRESHOLDS[pred_class]:
            x1, y1, x2, y2 = coordinates[i]
            color = colors[pred_class]
            
            # Draw Filled Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            # Draw Text
            label = LABELS[pred_class]
            cv2.putText(img, f"{label} {int(score*100)}%", (x1+5, y1+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 4. Blend Overlay and Display
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw Grid Lines for reference
    for r in range(1, ROWS): 
        cv2.line(img, (0, r*CELL_H), (IMG_WIDTH, r*CELL_H), (255, 255, 0), 1)
    for c in range(1, COLS): 
        cv2.line(img, (c*CELL_W, 0), (c*CELL_W, IMG_HEIGHT), (255, 255, 0), 1)

    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 10))
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

# %%
# --- TEST ON AN IMAGE ---
# Change the filename below to an image in your dataset_processed folder
test_image_name = "sample5.jpg" 
test_image_path = os.path.join(DATA_FOLDER, test_image_name)

if os.path.exists(test_image_path):
    predict_and_visualize(test_image_path, model)
else:
    print(f"⚠️ Please upload an image named '{test_image_name}' to '{DATA_FOLDER}' to test prediction.")

# %% [markdown]
# CONSOLIDATED CODE

# %%
# =========================================================
# IMPORTS
# =========================================================
import os, cv2, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from skimage.feature import hog, local_binary_pattern
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
TRAIN_IMAGE_FOLDER = "train_images1"
TEST_IMAGE_FOLDER  = "test_images"
CSV_FILE = "annotations.csv"

OUTPUT_ROOT = "output_results"
TRAIN_ANN_IMG_DIR = f"{OUTPUT_ROOT}/train_annotations_images"
TEST_ANN_IMG_DIR  = f"{OUTPUT_ROOT}/test_predictions_images"
PRED_NPY_DIR      = f"{OUTPUT_ROOT}/test_predictions_npy"
CSV_OUT_DIR       = f"{OUTPUT_ROOT}/group_08_output.csv"

for d in [TRAIN_ANN_IMG_DIR, TEST_ANN_IMG_DIR, PRED_NPY_DIR, CSV_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_FILENAME = "group_08_pickle.pkl"

IMG_WIDTH, IMG_HEIGHT = 800, 600
ROWS, COLS = 8, 8
CELL_W, CELL_H = IMG_WIDTH // COLS, IMG_HEIGHT // ROWS

LABELS = {0:"Background", 1:"Ball", 2:"Bat", 3:"Stump"}
COLORS = {1:(0,0,255), 2:(255,0,0), 3:(0,255,0)}

# =========================================================
# FEATURE FUNCTIONS
# =========================================================
def get_hog_features(img):
    return hog(img, orientations=9, pixels_per_cell=(16,16),
               cells_per_block=(2,2), block_norm="L2-Hys",
               transform_sqrt=True)

def get_color_histogram(hsv, bins=16):
    h = cv2.calcHist([hsv],[0],None,[bins],[0,180])
    s = cv2.calcHist([hsv],[1],None,[bins],[0,256])
    v = cv2.calcHist([hsv],[2],None,[bins],[0,256])
    cv2.normalize(h,h); cv2.normalize(s,s); cv2.normalize(v,v)
    return np.concatenate([h,s,v]).flatten()

def get_color_stats(hsv):
    mean,std = cv2.meanStdDev(hsv)
    return np.concatenate([mean,std]).flatten()

def get_lbp_features(gray):
    lbp = local_binary_pattern(gray,8,1,method="uniform")
    hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,11))
    hist = hist.astype("float")
    return hist/(hist.sum()+1e-7)

# =========================================================
# TRAIN DATA PREPARATION
# =========================================================
def prepare_training_data():
    df = pd.read_csv(CSV_FILE, encoding="latin1")
    X, y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(TRAIN_IMAGE_FOLDER, row["ImageFileName"])
        if not os.path.exists(img_path):
            continue

        img = cv2.resize(cv2.imread(img_path),(IMG_WIDTH,IMG_HEIGHT))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        cell = 0
        for r in range(ROWS):
            for c in range(COLS):
                cell += 1
                y1,y2 = r*CELL_H,(r+1)*CELL_H
                x1,x2 = c*CELL_W,(c+1)*CELL_W

                features = np.concatenate([
                    get_hog_features(gray[y1:y2,x1:x2]),
                    get_color_histogram(hsv[y1:y2,x1:x2]),
                    get_color_stats(hsv[y1:y2,x1:x2]),
                    get_lbp_features(gray[y1:y2,x1:x2])
                ])

                X.append(features)
                y.append(int(row[f"c{cell:02d}"]))

    X = np.array(X)
    y = np.array(y)

    np.save("X_train.npy", X)
    np.save("y_train.npy", y)

    print(f"✅ TOTAL TRAIN FEATURES: {X.shape[0]}")
    print(f"✅ FEATURES PER CELL: {X.shape[1]}")

    return X, y, df

# =========================================================
# TRAIN MODEL
# =========================================================
X_train, y_train, df_train = prepare_training_data()

weights = compute_sample_weight("balanced", y_train)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    tree_method="hist",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=weights)

with open(MODEL_FILENAME,"wb") as f:
    pickle.dump(model,f)

print("✅ Model trained and saved")

# =========================================================
# ANNOTATE TRAIN IMAGES (GROUND TRUTH)
# =========================================================
for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    img_path = os.path.join(TRAIN_IMAGE_FOLDER, row["ImageFileName"])
    if not os.path.exists(img_path):
        continue

    img = cv2.resize(cv2.imread(img_path),(IMG_WIDTH,IMG_HEIGHT))
    overlay = img.copy()

    cell = 0
    for r in range(ROWS):
        for c in range(COLS):
            cell += 1
            label = int(row[f"c{cell:02d}"])
            if label == 0:
                continue

            y1,y2 = r*CELL_H,(r+1)*CELL_H
            x1,x2 = c*CELL_W,(c+1)*CELL_W
            cv2.rectangle(overlay,(x1,y1),(x2,y2),COLORS[label],-1)

    cv2.addWeighted(overlay,0.4,img,0.6,0,img)
    cv2.imwrite(f"{TRAIN_ANN_IMG_DIR}/{row['ImageFileName']}", img)

print("✅ Annotated TRAIN images saved")

# =========================================================
# TEST + ANNOTATE TEST IMAGES
# =========================================================
csv_rows = []
X_test_all = []

for img_name in tqdm(os.listdir(TEST_IMAGE_FOLDER)):
    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    img_path = os.path.join(TEST_IMAGE_FOLDER,img_name)
    img = cv2.resize(cv2.imread(img_path),(IMG_WIDTH,IMG_HEIGHT))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    feats, coords = [], []

    for r in range(ROWS):
        for c in range(COLS):
            y1,y2 = r*CELL_H,(r+1)*CELL_H
            x1,x2 = c*CELL_W,(c+1)*CELL_W
            feats.append(np.concatenate([
                get_hog_features(gray[y1:y2,x1:x2]),
                get_color_histogram(hsv[y1:y2,x1:x2]),
                get_color_stats(hsv[y1:y2,x1:x2]),
                get_lbp_features(gray[y1:y2,x1:x2])
            ]))
            coords.append((x1,y1,x2,y2))

    feats = np.array(feats)
    X_test_all.append(feats)

    probs = model.predict_proba(feats)
    preds = np.argmax(probs, axis=1)

    np.save(f"{PRED_NPY_DIR}/{img_name}.npy", preds)

    overlay = img.copy()
    for i,p in enumerate(preds):
        if p != 0:
            x1,y1,x2,y2 = coords[i]
            cv2.rectangle(overlay,(x1,y1),(x2,y2),COLORS[p],-1)
            csv_rows.append([img_name,i+1,p,LABELS[p]])

    cv2.addWeighted(overlay,0.4,img,0.6,0,img)
    cv2.imwrite(f"{TEST_ANN_IMG_DIR}/{img_name}", img)

X_test = np.vstack(X_test_all)
np.save("X_test.npy", X_test)

print(f"✅ TOTAL TEST FEATURES: {X_test.shape[0]}")

# =========================================================
# SAVE CSV FILES
# =========================================================
pd.DataFrame(X_train).to_csv(f"{CSV_OUT_DIR}/X_train_features.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{CSV_OUT_DIR}/X_test_features.csv", index=False)

pd.DataFrame(
    csv_rows,
    columns=["image","cell","class_id","class_name"]
).to_csv(f"{CSV_OUT_DIR}/tgroup_08_output.csv", index=False)

print("🎯 PIPELINE COMPLETED SUCCESSFULLY")


# %%
import os
import cv2
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# =====================================================
# CONFIGURATION
# =====================================================
TRAIN_IMAGE_FOLDER = "train_images"
TEST_IMAGE_FOLDER  = "test_images"
CSV_FILE = "annotations.csv"

IMG_WIDTH, IMG_HEIGHT = 800, 600
ROWS, COLS = 8, 8
CELL_W, CELL_H = IMG_WIDTH // COLS, IMG_HEIGHT // ROWS

LABELS = {0:"Background",1:"Ball",2:"Bat",3:"Stumps"}
COLORS = {1:(0,0,255),2:(255,0,0),3:(0,255,0)}

OUTPUT_ROOT = "output_results"
TRAIN_ANN_IMG_DIR = f"{OUTPUT_ROOT}/train_annotations_images"
TEST_ANN_IMG_DIR  = f"{OUTPUT_ROOT}/test_predictions_images"
FEATURE_CSV_DIR   = f"{OUTPUT_ROOT}/features_csv"

for d in [OUTPUT_ROOT, TRAIN_ANN_IMG_DIR, TEST_ANN_IMG_DIR, FEATURE_CSV_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================
# FEATURE FUNCTIONS
# =====================================================
def get_hog_features(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16,16),
               cells_per_block=(2,2), block_norm='L2-Hys',
               transform_sqrt=True)

def get_color_histogram(hsv, bins=16):
    h = cv2.calcHist([hsv],[0],None,[bins],[0,180])
    s = cv2.calcHist([hsv],[1],None,[bins],[0,256])
    v = cv2.calcHist([hsv],[2],None,[bins],[0,256])
    cv2.normalize(h,h); cv2.normalize(s,s); cv2.normalize(v,v)
    return np.concatenate([h,s,v]).flatten()

def get_lbp(gray):
    lbp = local_binary_pattern(gray,8,1,'uniform')
    hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,11))
    hist = hist.astype("float")
    hist /= (hist.sum()+1e-6)
    return hist

# =====================================================
# DATASET PREPARATION (TRAIN)
# =====================================================
def prepare_training_data():
    df = pd.read_csv(CSV_FILE, encoding="latin1")

    X, y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(TRAIN_IMAGE_FOLDER, row["ImageFileName"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        cell = 0
        for r in range(ROWS):
            for c in range(COLS):
                cell += 1
                y1,y2 = r*CELL_H,(r+1)*CELL_H
                x1,x2 = c*CELL_W,(c+1)*CELL_W

                patch_g = gray[y1:y2,x1:x2]
                patch_h = hsv[y1:y2,x1:x2]

                feats = np.concatenate([
                    get_hog_features(patch_g),
                    get_color_histogram(patch_h),
                    get_lbp(patch_g)
                ])

                X.append(feats)
                y.append(int(row[f"c{cell:02d}"]))

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Total training samples: {len(X)}")
    print(f"✅ Features per cell: {X.shape[1]}")

    return X, y

# =====================================================
# TRAIN MODEL
# =====================================================
X, y = prepare_training_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

weights = compute_sample_weight("balanced", y_train)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train, sample_weight=weights)

print("🎯 Accuracy:", accuracy_score(y_test, model.predict(X_test)))

pickle.dump(model, open("xgb_model.pkl","wb"))

# =====================================================
# GRID PREDICTION FUNCTION
# =====================================================
def predict_image_grid(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Bad image: {img_path}")

    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    feats, coords = [], []

    for r in range(ROWS):
        for c in range(COLS):
            y1,y2 = r*CELL_H,(r+1)*CELL_H
            x1,x2 = c*CELL_W,(c+1)*CELL_W

            patch_g = gray[y1:y2,x1:x2]
            patch_h = hsv[y1:y2,x1:x2]

            feats.append(np.concatenate([
                get_hog_features(patch_g),
                get_color_histogram(patch_h),
                get_lbp(patch_g)
            ]))
            coords.append((x1,y1,x2,y2))

    preds = model.predict(np.array(feats))
    return preds, coords, img

# =====================================================
# PROCESS TRAIN & TEST + FINAL CSV (66 COLUMNS)
# =====================================================
rows = []

def process_folder(folder, split, out_dir):
    for img_name in tqdm(sorted(os.listdir(folder))):
        if not img_name.lower().endswith((".jpg",".png",".jpeg")):
            continue

        try:
            preds, coords, img = predict_image_grid(os.path.join(folder,img_name))
        except:
            continue

        row = {"ImageFileName":img_name,"dataset_split":split}
        for i,p in enumerate(preds):
            row[f"C{i+1:02d}"] = int(p)
        rows.append(row)

        overlay = img.copy()
        for i,p in enumerate(preds):
            if p!=0:
                x1,y1,x2,y2 = coords[i]
                cv2.rectangle(overlay,(x1,y1),(x2,y2),COLORS[p],-1)

        cv2.addWeighted(overlay,0.4,img,0.6,0,img)
        cv2.imwrite(f"{out_dir}/{img_name}",img)

process_folder(TRAIN_IMAGE_FOLDER,"Train",TRAIN_ANN_IMG_DIR)
process_folder(TEST_IMAGE_FOLDER,"Test",TEST_ANN_IMG_DIR)

# =====================================================
# SAVE FINAL CSV (66 COLUMNS)
# =====================================================
final_df = pd.DataFrame(rows)
final_df = final_df[["ImageFileName","dataset_split"] +
                    [f"C{i:02d}" for i in range(1,65)]]

final_df.to_csv(f"{OUTPUT_ROOT}/final_predictions.csv", index=False)
print("📄 Final 66-column CSV saved")



