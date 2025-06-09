## Feature Transformation and Visualization Script

---

### Overview

This repository contains a Python script designed to process and visualize vibration data from multiple sensors in both the time and frequency domains. The script applies various **feature transformations** to the raw data, allowing for an in-depth analysis of different signal characteristics.

The script loads data from `.lvm` files, applies a series of time-domain and frequency-domain feature transformations, and then generates plots for each transformed dataset. This helps in understanding how different statistical features and spectral properties of the vibration signals change across various samples.

---

### Features Implemented

The script includes functions for computing and applying the following features:

* **Amplitude**: Represents the maximum amplitude of the signal.
* **Decay**: Applies an inverse-time/frequency weighting to the signal.
* **Mean**: Calculates and sets the signal to its mean value.
* **Standard Deviation (std)**: Normalizes the signal by its mean and standard deviation.
* **Root Mean Square (RMS)**: Normalizes the signal by its RMS value.
* **Shape Factor (shape)**: Computes the ratio of RMS to the mean of the absolute values.
* **Kurtosis**: Measures the "tailedness" of the signal's probability distribution.
* **Skewness**: Measures the asymmetry of the signal's probability distribution.
* **Impulse Factor (impulse)**: Ratio of the peak value to the mean of the absolute values.
* **Crest Factor (crest)**: Ratio of the peak value to the RMS value.

Each feature has a corresponding function for both **time-domain** data and **frequency-domain (FFT)** data.

---

### Usage

1.  **Data Preparation**:
    * Place your time-domain vibration data files (e.g., `Sample_1.lvm`, `Sample_2.lvm`, ..., `Sample_19.lvm`) in the same directory as the script.
    * Ensure your frequency-domain FFT data files (e.g., `TestFFT_1.lvm`, `TestFFT_2.lvm`, ..., `TestFFT_19.lvm`) are also in the same directory.
    * The script expects these files to be CSV-like, with the first 23 rows skipped (header information) and data delimited by commas.
    * The data within these files should have at least four columns:
        * Column 0: Time or Frequency values.
        * Columns 1, 2, 3: Sensor acceleration/amplitude data.

2.  **Running the Script**:
    * Execute the Python script. It will automatically load the data, apply each feature transformation sequentially, and generate plots.

    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file.)

3.  **Output**:
    * For each feature, the script will generate two plots:
        * A **time-domain plot**: Named `vibration_[feature_name].png` and `vibration_[feature_name].pdf`.
        * A **frequency-domain plot**: Named `fft_[feature_name].png` and `fft_[feature_name].pdf`.
    * These plots will be saved in the same directory where the script is executed.

---

### Dependencies

This script requires the following Python libraries:

* `numpy`
* `matplotlib`
* `scipy`
* `IPython` (specifically for `IP.get_ipython().run_line_magic('reset', '-sf')` which clears the IPython console; it's not strictly necessary for script execution outside of an IPython environment).

You can install these dependencies using pip:

```bash
pip install numpy matplotlib scipy ipython


---
### Additional Notes for Dr. Fu and Dr. Downey


1. I used small values like 1e-8, to avoid division by zero
2. The functions will use an average acceleration computed between the 3 acceleration values





