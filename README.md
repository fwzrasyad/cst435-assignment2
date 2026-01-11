# CST435 Assignment 2: Parallel Image Processing System

This project implements a parallel image processing pipeline using Python. It applies a series of computationally intensive filters to the **Food-101 dataset** to benchmark performance differences between sequential and parallel execution paradigms.

## Group Members
* MUHAMMAD FAWWAZ RASYAD (160642)
* NOOR MOHAMMAD SOWAN (160235)
* MOHAMMAD HAZIQ SURMA (160800)
* ‘AUNI BINTI AHMAD (165152)


## Project Overview
The system processes a collection of images by applying five sequential filters:
1.  **Grayscale Conversion:** Converts RGB images to single-channel grayscale.
2.  **Gaussian Blur:** Applies a 3x3 kernel for noise reduction.
3.  **Edge Detection:** Uses the Sobel operator (X and Y gradients) to highlight edges.
4.  **Image Sharpening:** Enhances details using a custom sharpening kernel.
5.  **Brightness Adjustment:** Increases pixel intensity.

## Parallel Implementation
To analyze scalability and performance, the pipeline is implemented using two different Python parallel modules:
* **`multiprocessing`**: Uses `multiprocessing.Pool` for process-based parallelism.
* **`concurrent.futures`**: Uses `ProcessPoolExecutor` for high-level asynchronous execution.

## Installation & Setup

### 1. Prerequisites
* Python 3.x
* PIP (Python Package Installer)

### 2. Install Dependencies
Run the following command to install the required libraries (OpenCV and NumPy):
```bash
pip install -r requirements.txt
