import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
from PIL import Image, ImageTk, ImageDraw 
import csv
import pandas as pd 

# --- Configuration ---
TARGET_WIDTH = 800
TARGET_HEIGHT = 600
GRID_SIZE = 8
CELL_WIDTH = TARGET_WIDTH // GRID_SIZE  # 100 pixels
CELL_HEIGHT = TARGET_HEIGHT // GRID_SIZE # 75 pixels
BORDER_THICKNESS = 5 # Increased thickness for better visibility

# Annotation Labels and Values (Value: (Display Name, Overlay Color))
# Value corresponds to the label saved in the CSV: Ball=1, Bat=2, Stumps=3
ANNOTATION_MAP = {
    'Ball': (1, 'green'),
    'Bat': (2, 'red'),
    'Stumps': (3, 'blue'),
}

class ImageAnnotatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Cricket Image Feature Engineering Tool")
        
        self.input_folder = ""
        self.image_files = []
        self.current_image_index = 0
        self.current_image_path = None
        self.original_image_name = None
        self.current_processed_image = None # PIL Image object (800x600)
        self.current_annotations = [0] * (GRID_SIZE * GRID_SIZE) # Stores 64 values (0, 1, 2, 3)
        self.current_split = tk.StringVar(value='Train') 

        self.setup_ui()
        self.csv_filepath = 'annotations.csv'
        self.processed_dir = 'dataset_processed' 
        self.marked_dir = 'marked_dataset'       
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.marked_dir, exist_ok=True) 
        
        self.ensure_csv_header()

    def setup_ui(self):
        # --------------------- Controls Frame (Left) ---------------------
        controls_frame = ttk.Frame(self.master, padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 1. Folder Selection
        ttk.Label(controls_frame, text="1. Select Image Folder:").pack(pady=5, anchor='w')
        self.folder_label = ttk.Label(controls_frame, text="No folder selected", wraplength=200)
        self.folder_label.pack(fill='x', pady=5)
        ttk.Button(controls_frame, text="Browse Folder", command=self.browse_folder).pack(fill='x', pady=5)
        
        # Image Navigation
        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=10)
        self.status_label = ttk.Label(controls_frame, text="Status: Ready")
        self.status_label.pack(pady=5, anchor='w')
        
        self.nav_frame = ttk.Frame(controls_frame)
        self.nav_frame.pack(pady=10)
        self.prev_button = ttk.Button(self.nav_frame, text="< Previous", command=self.prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.nav_frame, text="Next >", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # --------------------- Annotation Controls (Center) ---------------------
        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(controls_frame, text="2. Select Object Type (Click cell to mark/unmark):").pack(pady=5, anchor='w')

        self.selected_object_name = tk.StringVar(value='Ball') # Default selection name
        self.object_type_frame = ttk.Frame(controls_frame)
        self.object_type_frame.pack(pady=5, fill='x')

        # Create radio buttons for Ball, Bat, Stumps
        for name, (_, color) in ANNOTATION_MAP.items():
            rb = ttk.Radiobutton(self.object_type_frame, text=f"{name} ({color.capitalize()})", 
                                 variable=self.selected_object_name, value=name)
            rb.pack(anchor='w', pady=2)
            
        # --------------------- Train/Test Split Selector ---------------------
        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(controls_frame, text="3. Select Data Split:").pack(pady=5, anchor='w')
        
        self.split_frame = ttk.Frame(controls_frame)
        self.split_frame.pack(pady=5, fill='x')
        
        ttk.Radiobutton(self.split_frame, text="Train", variable=self.current_split, value='Train').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.split_frame, text="Test", variable=self.current_split, value='Test').pack(side=tk.LEFT, padx=5)
        
        # --------------------- Actions ---------------------
        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(controls_frame, text="4. Actions:").pack(pady=5, anchor='w') 
        
        ttk.Button(controls_frame, text="Clear All Annotations on Current Image", command=self.clear_annotations).pack(fill='x', pady=5)
        
        # Save Button (Spacebar binding)
        ttk.Label(controls_frame, text="Press [Spacebar] to Save & Advance (Saves CSV & Marked Image).", font=('Arial', 10, 'bold')).pack(pady=10, anchor='w')

        # --------------------- Image Display Frame (Right) ---------------------
        image_frame = ttk.Frame(self.master, padding="10")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas_label = ttk.Label(image_frame, text="800 x 600 Image Annotation Canvas")
        self.canvas_label.pack(pady=5)
        
        self.canvas = tk.Canvas(image_frame, width=TARGET_WIDTH, height=TARGET_HEIGHT, bg='lightgray')
        self.canvas.pack(expand=True)
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # Bind spacebar to save and advance
        self.master.bind('<space>', lambda event: self.save_and_next())

    def ensure_csv_header(self):
        """Creates the CSV file with the required header if it doesn't exist."""
        if not os.path.exists(self.csv_filepath):
            feature_cols = [f'c0{i}' for i in range(1, 65)]
            header = ['image_name', 'Train/Test'] + feature_cols
            try:
                with open(self.csv_filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                print(f"Created new CSV file: {self.csv_filepath}")
            except IOError as e:
                messagebox.showerror("File Error", f"Could not create annotations.csv: {e}")

    def _reject_and_skip(self, reason_message):
        """Displays rejection message, clears canvas, and attempts to move to the next image."""
        self.canvas.delete("all")
        self.current_processed_image = None
        self.status_label.config(text=f"Image Rejected: {self.original_image_name}")
        messagebox.showwarning("Image Rejected", reason_message)
        
        # Manually advance the index and try to load the next one
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        else:
            messagebox.showinfo("Complete", "All images processed or rejected in this folder!")
            self.status_label.config(text="Status: Processing complete.")
            self.next_button.config(state=tk.DISABLED) # Ensure navigation is disabled

    def browse_folder(self):
        """Opens a dialog to select the folder containing images."""
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder = folder
            self.folder_label.config(text=f"Selected: {os.path.basename(folder)}")
            # Filter for common image file types
            self.image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
            self.current_image_index = 0
            
            if not self.image_files:
                self.status_label.config(text="Status: No images found.")
                self.next_button.config(state=tk.DISABLED)
                messagebox.showinfo("Info", "No images found in the selected folder. Supported types: .png, .jpg, .jpeg, .webp.")
            else:
                self.status_label.config(text=f"Status: Found {len(self.image_files)} images.")
                self.load_current_image() 
                self.next_button.config(state=tk.NORMAL)

    def load_current_image(self):
        """
        Loads, validates, processes (resizes/crops to 800x600), and displays the current image.
        """
        if not self.image_files or not (0 <= self.current_image_index < len(self.image_files)):
            self.canvas.delete("all")
            self.status_label.config(text="Status: All images processed or no images.")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return

        # Update file paths and name
        self.original_image_name = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.input_folder, self.original_image_name)
        self.status_label.config(text=f"Image {self.current_image_index + 1}/{len(self.image_files)}: {self.original_image_name}")

        try:
            # 1. Open and Validate Image Dimensions
            original_img = Image.open(self.current_image_path).convert("RGB")
            
            orig_w, orig_h = original_img.size
            
            # --- VALIDATION CHECK ---
            if orig_w < TARGET_WIDTH or orig_h < TARGET_HEIGHT:
                rejection_msg = (
                    f"Image '{self.original_image_name}' rejected.\n"
                    f"Dimensions found: {orig_w}x{orig_h}.\n"
                    f"Minimum dimension allowed is {TARGET_WIDTH}x{TARGET_HEIGHT}."
                )
                self._reject_and_skip(rejection_msg)
                return # Stop processing this image

            # 2. Resize the image to 800x600 (4:3) maintaining quality and aspect ratio (Processing continues only if valid)
            target_ratio = TARGET_WIDTH / TARGET_HEIGHT 

            if orig_w / orig_h > target_ratio:
                crop_h = orig_h
                crop_w = int(orig_h * target_ratio)
                left = (orig_w - crop_w) // 2
                top = 0
                right = left + crop_w
                bottom = crop_h
            else:
                crop_w = orig_w
                crop_h = int(orig_w / target_ratio)
                left = 0
                top = (orig_h - crop_h) // 2
                right = crop_w
                bottom = top + crop_h

            cropped_img = original_img.crop((left, top, right, bottom))
            
            resized_img = cropped_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
            self.current_processed_image = resized_img
            
            processed_image_path = os.path.join(self.processed_dir, self.original_image_name)
            resized_img.save(processed_image_path)
            print(f"Saved processed 800x600 image to {processed_image_path}")

            # 3. Load existing annotations and split status or start fresh
            self.load_annotations()
            
            # 4. Display the processed image with grid and numbers
            self.display_image()
            
            # 5. Update nav buttons state
            self.prev_button.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.NORMAL if self.current_image_index < len(self.image_files) - 1 else tk.DISABLED)

        except Exception as e:
            self.canvas.delete("all")
            self.status_label.config(text=f"Error loading image: {self.original_image_name}")
            print(f"Error loading image {self.current_image_path}: {e}")
            # If any other error occurs (like corrupt file), skip it
            self._reject_and_skip(f"An error occurred while loading or processing image '{self.original_image_name}': {e}")
            return 


    def load_annotations(self):
        """Loads the most recent annotations and split status for the current image from the CSV if they exist."""
        self.current_annotations = [0] * (GRID_SIZE * GRID_SIZE)
        self.current_split.set('Train') 
        
        try:
            if os.path.exists(self.csv_filepath) and os.path.getsize(self.csv_filepath) > 0:
                df = pd.read_csv(self.csv_filepath, dtype={'image_name': str}, index_col=False)
                
                image_rows = df[df['image_name'] == self.original_image_name]
                
                if not image_rows.empty:
                    latest_row = image_rows.iloc[-1]
                    
                    if 'Train/Test' in latest_row:
                         self.current_split.set(latest_row['Train/Test'])

                    feature_cols = [f'c0{i}' for i in range(1, 65)]
                    features = latest_row[feature_cols].tolist()
                    self.current_annotations = [int(f) for f in features]
                    print(f"Loaded existing annotations for {self.original_image_name}.")
                    return

            print(f"No existing annotations found for {self.original_image_name}. Starting fresh.")
                
        except pd.errors.EmptyDataError:
            print("Annotation CSV is empty.")
            self.ensure_csv_header()
        except FileNotFoundError:
            self.ensure_csv_header()
        except Exception as e:
            print(f"Error loading annotations from CSV: {e}")
            self.ensure_csv_header() 

    def display_image(self):
        """Draws the image, grid, cell numbers, and annotation overlays on the canvas."""
        if not self.current_processed_image:
            return

        self.tk_image = ImageTk.PhotoImage(self.current_processed_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Draw Grid and Cell Numbers
        for i in range(GRID_SIZE): # Row index (0-7)
            for j in range(GRID_SIZE): # Column index (0-7)
                cell_index = i * GRID_SIZE + j  # 0 to 63
                cell_number = cell_index + 1    # 1 to 64
                
                x1, y1 = j * CELL_WIDTH, i * CELL_HEIGHT
                x2, y2 = x1 + CELL_WIDTH, y1 + CELL_HEIGHT
                
                # --- Drawing the Base Grid on Canvas ---
                # Vertical lines
                self.canvas.create_line(x1, 0, x1, TARGET_HEIGHT, fill="white", width=1)
                # Horizontal lines
                self.canvas.create_line(0, y1, TARGET_WIDTH, y1, fill="white", width=1)

                # --- Drawing Cell Numbers ---
                self.canvas.create_text(x1 + 5, y1 + 5, anchor=tk.NW, text=str(cell_number), 
                                        fill="yellow", font=('Arial', 10, 'bold'))

                # Apply Annotation Overlay AND Thickened Border
                annotation_value = self.current_annotations[cell_index]
                if annotation_value != 0:
                    # Find the color based on the value (1, 2, or 3)
                    color_name = next(color for value, color in ANNOTATION_MAP.values() if value == annotation_value)
                    
                    # Create a semi-transparent filled rectangle with a highly visible border
                    self.canvas.create_rectangle(x1, y1, x2, y2, 
                                                fill=color_name, 
                                                stipple='gray25', # Semi-transparent effect for fill
                                                outline=color_name, # Use annotation color for border
                                                width=BORDER_THICKNESS) # Uses the increased BORDER_THICKNESS constant

    def handle_click(self, event):
        """Handles cell selection on canvas click."""
        if not self.current_processed_image:
            return

        # Determine which cell was clicked
        j = event.x // CELL_WIDTH  # Column index (0-7)
        i = event.y // CELL_HEIGHT # Row index (0-7)
        
        if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
            cell_index = i * GRID_SIZE + j
            
            selected_name = self.selected_object_name.get()
            new_annotation_value = ANNOTATION_MAP[selected_name][0]
            
            current_value = self.current_annotations[cell_index]
            
            if new_annotation_value == current_value:
                # If clicking the currently selected object type on an already marked cell, unmark it (set to 0)
                self.current_annotations[cell_index] = 0
            else:
                # Otherwise, apply the new annotation value directly
                self.current_annotations[cell_index] = new_annotation_value

            self.display_image() # Redraw the canvas with updated annotations

    def clear_annotations(self):
        """Clears all annotations for the current image."""
        if messagebox.askyesno("Clear Annotations", "Are you sure you want to clear ALL annotations for the current image?"):
            self.current_annotations = [0] * (GRID_SIZE * GRID_SIZE)
            self.display_image()

    def draw_overlay_and_save(self):
        """
        Creates a new PIL image by drawing the grid, cell numbers, and annotations
        onto the current processed image and saves it to the marked_dataset folder.
        """
        if not self.current_processed_image:
            return

        marked_img = self.current_processed_image.copy()
        draw = ImageDraw.Draw(marked_img, 'RGBA') 

        # Define colors (R, G, B, Alpha) - Alpha 0 to 255
        COLOR_MAP_FILL = {
            1: (0, 255, 0, 90),   # Green (Ball) - Semi-transparent fill
            2: (255, 0, 0, 90),   # Red (Bat)
            3: (0, 0, 255, 90),   # Blue (Stumps)
        }
        COLOR_MAP_BORDER = {
            1: (0, 255, 0, 255),   # Green (Ball) - Opaque border
            2: (255, 0, 0, 255),   # Red (Bat)
            3: (0, 0, 255, 255),   # Blue (Stumps)
        }
        
        # Draw Annotations and Grid
        for i in range(GRID_SIZE): 
            for j in range(GRID_SIZE):
                cell_index = i * GRID_SIZE + j
                
                x1, y1 = j * CELL_WIDTH, i * CELL_HEIGHT
                x2, y2 = x1 + CELL_WIDTH, y1 + CELL_HEIGHT

                # 1. Apply Annotation Overlay (Semi-transparent rectangle with Opaque Border)
                annotation_value = self.current_annotations[cell_index]
                if annotation_value != 0:
                    fill_color = COLOR_MAP_FILL.get(annotation_value)
                    border_color = COLOR_MAP_BORDER.get(annotation_value)
                    
                    if fill_color and border_color:
                        # Draw the rectangle with semi-transparent fill and opaque, thickened border
                        draw.rectangle([(x1, y1), (x2, y2)], 
                                       fill=fill_color, 
                                       outline=border_color, 
                                       width=BORDER_THICKNESS) # Uses the increased BORDER_THICKNESS constant
                
                # 2. Draw Base Grid Lines (White)
                line_color = (255, 255, 255, 255) # White, opaque
                draw.line([(x1, 0), (x1, TARGET_HEIGHT)], fill=line_color, width=1) # Vertical
                draw.line([(0, y1), (TARGET_WIDTH, y1)], fill=line_color, width=1)   # Horizontal
        
        # Save the marked image
        marked_image_path = os.path.join(self.marked_dir, self.original_image_name)
        try:
            # Convert from RGBA to RGB for saving (handles transparency by blending with white)
            if marked_img.mode == 'RGBA':
                background = Image.new('RGB', marked_img.size, (255, 255, 255))
                background.paste(marked_img, mask=marked_img.split()[3]) 
                marked_img = background
            
            marked_img.save(marked_image_path)
            print(f"Saved marked image to {marked_image_path}")
        except Exception as e:
            print(f"Error saving marked image: {e}")
            messagebox.showerror("Save Error", f"Failed to save marked image: {e}")


    def save_and_next(self):
        """
        Saves the current image features and split status to CSV in append mode,
        saves the marked image, and moves to the next image.
        """
        if not self.current_image_path or not self.current_processed_image:
            # Prevent saving if the current image was rejected
            print("Skipping save: No valid image loaded.")
            return
            
        # --- Save the visually marked image ---
        self.draw_overlay_and_save()

        # Check if file exists AND is empty/just created to determine if header needs to be written
        write_header = not os.path.exists(self.csv_filepath) or os.path.getsize(self.csv_filepath) == 0

        # 1. Prepare Data in the correct order: image_name, Train/Test, c_1...
        data = {
            'image_name': self.original_image_name,
            'Train/Test': self.current_split.get()
        }
        # Map annotation list (0-63) to feature columns (c01 to c064)
        data.update({f'c0{i+1}': self.current_annotations[i] for i in range(64)})

        # Ensure column order matches the desired structure: image_name, Train/Test, c01...
        feature_cols = [f'c0{i}' for i in range(1, 65)]
        columns_order = ['image_name', 'Train/Test'] + feature_cols

        # 2. Save Features to CSV (Always Append)
        try:
            new_row_df = pd.DataFrame([data], columns=columns_order)
            
            # Use to_csv with mode='a' (append). 
            new_row_df.to_csv(self.csv_filepath, mode='a', header=write_header, index=False)
            
            print(f"Appended new annotations (Split: {self.current_split.get()}) for {self.original_image_name} to {self.csv_filepath}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save features to CSV: {e}")
            return
            
        # 3. Move to the next image
        self.next_image()

    def next_image(self):
        """Advances to the next image in the list."""
        # If we are already at the last image, don't increment
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        elif self.image_files:
            # Last image reached
            messagebox.showinfo("Complete", "All images have been processed or rejected in this folder!")
            self.status_label.config(text="Status: Processing complete.")
        else:
            self.status_label.config(text="Status: No images to process.")

    def prev_image(self):
        """Goes back to the previous image in the list."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = ImageAnnotatorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"A critical error occurred: {e}")
        messagebox.showerror("Critical Error", f"The application failed to start. Ensure Python and the required libraries (Pillow, pandas) are installed. Error: {e}")
