# Welcome to WHOLISTIC-registration (vmsr branch)
**A fast, accurate, and non-rigid image registration method for Whole-Body cellular activity Imaging.**

> **Note**: This is the `vmsr` development branch with additional features and improvements.


## Introduction
WHOLISTIC Registration is a method designed to correct non-rigid motion caused by skeletal and smooth muscle contractions. This enables precise cellular activity analysis and motion analysis for fluorescent data. The method utilizes patch-wise iterative modified optical flow with an image pyramid to achieve high flexibility and robustness.

Below are examples showcasing the results of WBI Registration:

-**Input (Left)**: Raw video

-**Output (Right)**: Motion-corrected video

<p align="center">
  <b>
    Example #1
  </b>
</p>

https://github.com/user-attachments/assets/871dfb15-49a5-47de-8b18-878880605e74

<p align="center">
  <b>
    Example #2
  </b>
</p>


https://github.com/user-attachments/assets/5fb45eca-02a1-4226-8208-2f6dbd6171a3

### Advantage
- **Accurate non-rigid registration**: Delivers precise motion correction.
- **Fast and parallelizable**: Optimized for GPU acceleration.
- **Flexible masking**: Allows users to ignore unwanted regions.
- **Robust to noise**: Performs well even with noisy fluorescent data.

### Suitable Data for WBI Registration
WBI Registration assumes that the template image and the moving image have similar intensities (or change gradually). Thus, At least one channel of your data should avoid fast-changing activities, such as Calcium activity signals, to ensure optimal results.


## Requirements
### OS Requirements
This package is supported for *Linux* and *Windows*. The package has been tested on the following systems:
+ Linux: Ubuntu 24.04


### Hardware Requirements
- A discrete GPU with sufficient memory is recommended for acceleration.
