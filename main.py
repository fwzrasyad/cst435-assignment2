import cv2
import numpy as np
import os
import time
import multiprocessing
import concurrent.futures
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = 'food_data'  # Folder containing your source images
OUTPUT_FOLDER = 'processed_images'
NUM_IMAGES_TO_PROCESS = 1000  # Set to 1000 for the final report benchmark

# ==========================================
# IMAGE FILTERS
# ==========================================
def apply_filters(image_path, output_dir):
    try:
        filename = os.path.basename(image_path)
        img = cv2.imread(str(image_path))
        
        if img is None:
            return f"Failed to load {filename}"

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur (3x3)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 3. Edge Detection (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # 4. Image Sharpening
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1], 
                                      [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)

        # 5. Brightness Adjustment
        brightness = cv2.convertScaleAbs(img, alpha=1, beta=50)

        # Save the final processed image
        # Using the 'sharpened' version as the output example
        cv2.imwrite(os.path.join(output_dir, f"processed_{filename}"), sharpened)
        
        return None # Success
    except Exception as e:
        return str(e)

# ==========================================
# BENCHMARK FUNCTIONS
# ==========================================

def run_sequential(image_paths):
    print("Running sequential version...")
    start = time.time()
    for path in image_paths:
        apply_filters(path, OUTPUT_FOLDER)
    return time.time() - start

def run_multiprocessing(image_paths, num_workers):
    print(f"Running multiprocessing with {num_workers} processes...")
    start = time.time()
    # We use starmap to pass multiple arguments (path, output_dir)
    tasks = [(path, OUTPUT_FOLDER) for path in image_paths]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(apply_filters, tasks)
        
    return time.time() - start

def run_futures(image_paths, num_workers):
    print(f"Running futures with {num_workers} workers...")
    start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(apply_filters, path, OUTPUT_FOLDER) for path in image_paths]
        # Wait for all to complete
        concurrent.futures.wait(futures)
        
    return time.time() - start

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # Setup output directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Get images
    all_images = [str(p) for p in Path(INPUT_FOLDER).glob("**/*.jpg")]

    # Select subset of images
    if len(all_images) >= NUM_IMAGES_TO_PROCESS:
        images = all_images[:NUM_IMAGES_TO_PROCESS]
    else:
        print(f"Warning: Only found {len(all_images)} images. Processing all of them.")
        images = all_images

    print(f"Processing {len(images)} images for benchmark...\n")

    # 1. Sequential Run (Baseline)
    seq_time = run_sequential(images)
    print(f"Sequential time: {seq_time:.4f} seconds\n")

    worker_counts = [2, 4, 8]

    # 2. Multiprocessing Module
    print("--- Multiprocessing Module ---")
    for w in worker_counts:
        t = run_multiprocessing(images, w)
        speedup = seq_time / t
        efficiency = speedup / w
        print(f"Time: {t:.4f}s | Speedup: {speedup:.2f} | Efficiency: {efficiency:.2f}")

    print("\n")

    # 3. Concurrent Futures
    print("--- Concurrent Futures ---")
    for w in worker_counts:
        t = run_futures(images, w)
        speedup = seq_time / t
        efficiency = speedup / w
        print(f"Time: {t:.4f}s | Speedup: {speedup:.2f} | Efficiency: {efficiency:.2f}")

    print("\nBenchmark Complete.")