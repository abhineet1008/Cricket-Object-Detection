# %% [markdown]
# # Exploratory Data Analysis
# 1. Load Entire Image data Set
# 2. Load annotated CSV file based on the labelling conducted for the images.
# 3. Match all teh CSV and Image file and print values to check if it is successfully loaded
# 4.  Train and Test split of the data

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. Configuration
# ==========================================
csv_file_path = "C:\\Users\\aysar\\Downloads\\Attempt_4\\annotations_4.csv"
image_folder_path = "C:\\Users\\aysar\\Downloads\\Attempt_4\\dataset_processed"

# ==========================================
# 2. Read the CSV Data (With Encoding Fix)
# ==========================================
try:
    # We use 'ISO-8859-1' to fix the UnicodeDecodeError
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    print(f"Successfully loaded CSV with {len(df)} rows.")
    print("First 5 rows of data:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    exit()
except UnicodeDecodeError:
    print("ISO-8859-1 failed. Trying 'cp1252'...")
    df = pd.read_csv(csv_file_path, encoding='cp1252')

# ==========================================
# 3. Split Data (80% Train, 20% Test)
# ==========================================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print("-" * 30)
print(f"Training set size: {len(train_df)} images (80%)")
print(f"Testing set size:  {len(test_df)} images (20%)")
print("-" * 30)

# ==========================================
# 4. Sanity Check (Check if images exist)
# ==========================================
# We check if the column 'image_name' exists, then check the files
if 'image_name' in df.columns:
    missing_files = []
    # Check first 5 images as a sample
    for index, row in df.head(5).iterrows(): 
        img_name = str(row['image_name'])
        img_path = os.path.join(image_folder_path, img_name)
        
        if not os.path.exists(img_path):
            missing_files.append(img_name)
            
    if missing_files:
        print(f"Warning: Could not find these sample images in the folder:")
        print(missing_files)
        print(f"Looking in: {image_folder_path}")
    else:
        print("Sanity check passed: Sample images found in the directory.")
else:
    print(f"Note: Column 'image_name' not found. Your columns are: {list(df.columns)}")

# %% [markdown]
# # Conduct further data analysis to understand the existing raw data 
# 1. Identify, if there is any data imbalance.
# 2. Check for any outliers in the image (using box plot)
# 3. Plot the frequency of location of object (Bat, Ball and Stumps) in images.
# ## Analysis based on output :
# This box plot visualizes the size distribution of each object class in our dataset. specifically measuring how many "grid cells" each object occupies per image. Since we are using an 8x8 grid, the maximum possible value on the Y-axis is 64.
# 1. The "Background" Dominance 
# •	In almost every image, the "Background" class occupies the vast majority (about 90%) of the 64 available grid cells. This is expected in cricket footage, where most of the frame is grass, pitch, or sky.
# 2. Ball Anomaly
# •	The orange box in box plot is compressed into a flat line at the very bottom (near 0–1). This indicates that in the majority of the images, the ball occupies extremely little space—likely just 1 or 2 grid cells.
# •	We have many "outlier" circles extending all the way up to 60, as there are extreme close-up shots of a ball in training data.
# 3. Bat and Stump (Blue and Pink Boxes)
# •	Like the ball, these objects typically occupy a small portion of the image (medians are low, between 2–5 cells).
# •	The "Stump" class (pink) has a slightly taller box and longer whisker than the "Bat" or "Ball," suggesting that stumps consistently take up a bit more vertical space or appear in larger groups than the ball does.
# ## Key Takeaways for our training Model
# 1.	Extreme Imbalance: Our model has to face a dataset where ~90% of the inputs are "Background." Without a balanced weight approach for any tree based modelling, the model might learn to just predict "Background" for everything to achieve high accuracy while failing to detect the actual objects.
# 

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
csv_path = "C:\\Users\\aysar\\Downloads\\Attempt_4\\annotations_4.csv"
output_dir = "C:\\Users\\aysar\\Downloads\\Attempt_4\\multiclass_reports"

# !!! IMPORTANT !!! 
# Update this to match the numbers inside your grid cells (c01-c64)
CLASS_MAP = {
    0: 'Background',
    1: 'Ball',
    2: 'Bat',
    3: 'Stump'
}

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================
print("--- Loading Data ---")
try:
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    df.columns = df.columns.str.strip() # Clean column names
    print(f"✅ Loaded {len(df)} rows.")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Identify grid columns
grid_cols = [f'c{i:02d}' for i in range(1, 65)]
missing_cols = [c for c in grid_cols if c not in df.columns]

if missing_cols:
    print(f"❌ Error: Missing grid columns: {missing_cols}")
    exit()

# ==========================================
# 3. BOX PLOT (Class Distribution per Image)
# ==========================================
print("\n--- Generating Box Plot ---")

# We need to count how many cells belong to each class for every single image
class_counts = {name: [] for name in CLASS_MAP.values()}

for _, row in df.iterrows():
    # Get values for this image (row)
    grid_values = row[grid_cols].values
    
    # Count occurrences of each class ID
    counts = pd.Series(grid_values).value_counts()
    
    for class_id, class_name in CLASS_MAP.items():
        # Get count, default to 0 if not found
        count = counts.get(class_id, 0)
        class_counts[class_name].append(count)

# Convert to DataFrame for plotting
df_counts = pd.DataFrame(class_counts)

# Plotting
plt.figure(figsize=(10, 6))
# We melt the dataframe to make it suitable for seaborn boxplot
df_melted = df_counts.melt(var_name='Object Class', value_name='Grid Cells Occupied')

sns.boxplot(x='Object Class', y='Grid Cells Occupied', data=df_melted, palette="Set2")
plt.title('Distribution of Object Sizes (Grid Cell Count per Image)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

save_path = os.path.join(output_dir, 'box_plot_class_distribution.png')
plt.savefig(save_path)
print(f"✅ Saved Box Plot: {save_path}")

# ==========================================
# 4. HEATMAPS (Spatial Distribution)
# ==========================================
print("\n--- Generating Heatmaps ---")

# Filter out 'Background' for heatmaps as requested (only Bat, Ball, Stump)
target_classes = {k: v for k, v in CLASS_MAP.items() if v != 'Background'}

# Create a figure with subplots (1 row, N columns)
fig, axes = plt.subplots(1, len(target_classes), figsize=(6 * len(target_classes), 5))

# If there is only one target class, axes is not a list, so we make it one
if len(target_classes) == 1:
    axes = [axes]

for ax, (class_id, class_name) in zip(axes, target_classes.items()):
    print(f"   Calculating heatmap for: {class_name}...")
    
    # Initialize 8x8 grid with zeros
    heatmap_accum = np.zeros(64)
    
    # Sum up positions across all images
    matches = (df[grid_cols] == class_id).sum()
    
    # Reshape into 8x8
    heatmap_data = matches.values.reshape(8, 8)
    
    # Plot
    sns.heatmap(heatmap_data, ax=ax, cmap="Reds", annot=True, fmt='g', cbar=True)
    ax.set_title(f'{class_name} Location Frequency')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.tight_layout()
save_path_heat = os.path.join(output_dir, 'heatmaps_bat_ball_stump.png')
plt.savefig(save_path_heat)
print(f"✅ Saved Heatmaps: {save_path_heat}")

# ==========================================
# 5. GLOBAL CLASS IMBALANCE (NEW SECTION)
# ==========================================
print("\n--- Generating Global Class Balance Report ---")

# Sum totals from the previously calculated per-image counts
total_counts = {k: sum(v) for k, v in class_counts.items()}
total_cells = sum(total_counts.values())

print(f"Total Grid Cells Processed: {total_cells}")
print("-" * 45)
print(f"{'Class':<15} | {'Count':<10} | {'Percentage':<10}")
print("-" * 45)

for class_name, count in total_counts.items():
    percentage = (count / total_cells) * 100
    print(f"{class_name:<15} | {count:<10} | {percentage:.2f}%")
print("-" * 45)

# Calculate Foreground vs Background ratio
background_count = total_counts['Background']
foreground_count = total_cells - background_count

if foreground_count > 0:
    ratio = background_count / foreground_count
    print(f"\nSummary:")
    print(f"Background: {background_count} ({background_count/total_cells:.2%})")
    print(f"Foreground (Ball+Bat+Stump): {foreground_count} ({foreground_count/total_cells:.2%})")
    print(f"Ratio (Bg : Fg) -> {ratio:.1f} : 1")
else:
    print("\nWarning: No foreground objects found!")

# Plotting the imbalance
plt.figure(figsize=(8, 6))
# Using a Logarithmic scale because background is usually 100x larger than others
bars = plt.bar(total_counts.keys(), total_counts.values(), color=['gray', 'red', 'blue', 'orange'])

# Add counts on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/total_cells:.1%})',
             ha='center', va='bottom', fontsize=9)

plt.title('Total Data Volume: Background vs Objects (Log Scale)')
plt.ylabel('Total Grid Cells (Log Scale)')
plt.yscale('log') # Using log scale so the small bars are actually visible
plt.grid(axis='y', linestyle='--', alpha=0.3)

save_path_imb = os.path.join(output_dir, 'global_class_imbalance.png')
plt.savefig(save_path_imb)
print(f"✅ Saved Imbalance Chart: {save_path_imb}")

print("\n--- Analysis Complete ---")




# %%
#print number of features extracted from each image
import pandas as pd
import os
# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these paths if needed
csv_file_path = r"C:\Users\aysar\Downloads\Attempt_4\output_fe_4\final_features.csv"
image_folder_path = r"C:\Users\aysar\Downloads\Attempt_4\dataset_processed"
# ==========================================
# 2. Read the CSV Data
# ==========================================

df = pd.read_csv(csv_file_path)
print(f"Total number of images with extracted features: {len(df)}")
print(f"Number of features extracted from each image: {len(df.columns) - 1
} (excluding image identifier column)")
print("Feature columns:")
print(df.columns[1:].tolist())  # Exclude the first column if it's an identifier

# %% [markdown]
# # Extract features to plot PCA and t-SNE

# %%
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import hog, local_binary_pattern

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these paths to match your folders
image_folder = "C:\\Users\\aysar\\Downloads\\Attempt_4\\dataset_processed"
input_csv_path = "C:\\Users\\aysar\\Downloads\\Attempt_4\\annotations_4.csv"
output_csv_path = "C:\\Users\\aysar\\Downloads\\Attempt_4\\output_fe_4\\final_features.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Image Settings
IMG_W, IMG_H = 800, 600
ROWS, COLS = 8, 8  # 8x8 Grid

# ==========================================
# 2. YOUR FEATURE EXTRACTORS
# ==========================================
def get_hog_features(gray_img):
    """Extracts Histogram of Oriented Gradients (Shape)"""
    try:
        # settings: 9 orientations, 16x16 pixels per cell, 2x2 cells per block
        features = hog(gray_img, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        return features
    except:
        # Fallback for edge cases (e.g. image too small)
        return np.zeros(1)

def get_color_histogram(hsv_img, bins=16):
    """Extracts Color Histograms for Hue, Saturation, Value"""
    # Calculate histograms for H, S, and V channels
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

# ==========================================
# 3. MASTER EXTRACTOR FUNCTION
# ==========================================
def extract_cell_features(cell_bgr):
    """
    Combines all your extractors into one feature vector for a single grid cell.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return None

    # Prepare inputs
    cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    cell_hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    
    features = []
    
    # 1. HOG
    hog_feats = get_hog_features(cell_gray)
    features.extend(hog_feats)
    
    # 2. Color Histogram
    color_hist = get_color_histogram(cell_hsv)
    features.extend(color_hist)
    
    # 3. Color Stats
    color_stats = get_color_stats(cell_hsv)
    features.extend(color_stats)
    
    # 4. LBP
    lbp_feats = get_lbp_features(cell_gray)
    features.extend(lbp_feats)
    
    return np.array(features)

# ==========================================
# 4. MAIN PROCESSING LOOP
# ==========================================
def process_dataset():
    print("--- Loading Annotations ---")
    try:
        df = pd.read_csv(input_csv_path, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Identify grid columns (c01 to c64)
    grid_cols = [f'c{i:02d}' for i in range(1, 65)]
    
    # Prepare list to store extracted data
    extracted_data = []
    
    print(f"--- Processing {len(df)} images ---")
    
    for idx, row in df.iterrows():
        fname = row['image_name']
        img_path = os.path.join(image_folder, fname)
        
        # Check file existence
        if not os.path.exists(img_path):
            if idx < 5: print(f"⚠️ Skipped missing file: {fname}")
            continue
            
        # Read Image
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Resize to standard dimensions
        img = cv2.resize(img, (IMG_W, IMG_H))
        
        # Calculate grid cell sizes
        dy, dx = IMG_H // ROWS, IMG_W // COLS
        
        # Get labels for this image
        try:
            labels = row[grid_cols].values
        except KeyError:
            print(f"⚠️ Skipped {fname} due to missing grid columns.")
            continue
            
        # --- Grid Loop ---
        cell_count = 0
        for r in range(0, IMG_H, dy):
            for c in range(0, IMG_W, dx):
                # Crop the grid cell
                cell_img = img[r:r+dy, c:c+dx]
                
                # Extract features using your logic
                feats = extract_cell_features(cell_img)
                
                if feats is not None:
                    # Create a dictionary record
                    record = {
                        'image_name': fname,
                        'cell_id': cell_count + 1,
                        'label': labels[cell_count]
                    }
                    
                    # Add features as individual columns (f_0, f_1, etc.)
                    for i, val in enumerate(feats):
                        record[f'f_{i}'] = val
                        
                    extracted_data.append(record)
                
                cell_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1} / {len(df)} images...")

    # Save to CSV
    print("--- Saving Data ---")
    if extracted_data:
        df_final = pd.DataFrame(extracted_data)
        df_final.to_csv(output_csv_path, index=False)
        print(f"✅ Success! Features saved to: {output_csv_path}")
        print(f"   Total Samples (Grid Cells): {len(df_final)}")
        print(f"   Feature Vector Size: {len(df_final.columns) - 3}") # Subtract meta columns
    else:
        print("❌ Error: No data was extracted. Check image paths.")

if __name__ == "__main__":
    process_dataset()

# %% [markdown]
# # Plot and check PCA and t-SNE plots
# 1. It is observed that there is a blob like image in PCA and no clustering observed in t-SNE plots
# 2. Tried addiitonal handcrafteed images but could not achieve better output
# 3. Expected that non-linear models such as XGBoost, Random Forrest will make better guess

# %%
import pandas as pd
import numpy as np
import matplotlib
# Force Matplotlib to save files without opening windows
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input should match the output from the previous feature extraction script
input_csv = r"C:\Users\aysar\Downloads\Attempt_4\output_fe_4\final_features.csv"
output_dir = r"C:\Users\aysar\Downloads\Attempt_4\visualizations"

MAX_SAMPLES = 10000 
LABEL_MAP = {0: 'Background', 1: 'Ball', 2: 'Bat', 3: 'Stumps'}

# Visualization Colors
COLORS = {0: 'green', 1: 'red', 2: 'blue', 3: 'orange'} 
ALPHAS = {0: 0.3, 1: 1.0, 2: 1.0, 3: 1.0}
SIZES =  {0: 10,  1: 40,  2: 40,  3: 40}

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_data(file_path):
    print(f"--- Loading Data from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("   (Run the feature extraction script first!)")
        return None, None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ CSV Read Error: {e}")
        return None, None
        
    # Check for empty file
    if len(df) == 0:
        print("❌ CSV is empty.")
        return None, None

    # Dynamically find feature columns (they start with 'f_')
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    
    if not feature_cols:
        print("❌ No feature columns (f_0, f_1...) found!")
        print("   Check if the previous script ran correctly.")
        return None, None

    print(f"   Found {len(feature_cols)} feature dimensions.")
    print(f"   Found {len(df)} total samples.")

    X = df[feature_cols].values
    y = df['label'].values
    
    return X, y

# ==========================================
# 3. VISUALIZATION PIPELINE
# ==========================================
def run_visualization():
    # 1. Load
    X, y = load_data(input_csv)
    if X is None: return

    # 2. Downsample (if too large)
    if len(X) > MAX_SAMPLES:
        print(f"   Downsampling to {MAX_SAMPLES} samples for speed...")
        indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub, y_sub = X, y

    # 3. Standardize (Crucial for PCA/t-SNE)
    print("   Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    # 4. PCA
    print("   Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    var_exp = np.sum(pca.explained_variance_ratio_)
    
    # 5. t-SNE
    print("   Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)

    # 6. Plotting
    print("   Generating Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')

    def plot_scatter(ax, data, title):
        ax.set_facecolor('white')
        
        for label_id, label_name in LABEL_MAP.items():
            indices = (y_sub == label_id)
            if np.sum(indices) > 0:
                ax.scatter(
                    data[indices, 0], data[indices, 1], 
                    c=COLORS.get(label_id, 'black'), 
                    label=label_name, 
                    alpha=ALPHAS.get(label_id, 0.5), 
                    s=SIZES.get(label_id, 20),
                    edgecolors='black', linewidth=0.5
                )
        ax.set_title(title, color='black', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)

        # RED LEGEND STYLE
        legend = ax.legend(loc='upper right', frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('#D32F2F') # Red Background
        frame.set_edgecolor('black')
        frame.set_alpha(1.0)
        for text in legend.get_texts():
            text.set_color('white') # White Text
            text.set_weight('bold')

    plot_scatter(axes[0], pca_result, f"PCA (Linear)\nExplained Variance: {var_exp:.2f}")
    plot_scatter(axes[1], tsne_result, f"t-SNE (Non-Linear)")

    plt.suptitle(f"Data Assessment: Handcrafted Features (N={len(X_sub)})", fontsize=16, color='black')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "data_assessment_pca_tsne.png")
    plt.savefig(save_path)
    print(f"✅ Plot saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    run_visualization()

# %%


# %%




