```markdown
# Classical Motion Detection & Background Modeling

A from-scratch implementation of a motion detection pipeline using classical Computer Vision techniques. This project focuses on the fundamental mathematics of image processing, implementing core algorithms like Mahalanobis distance, morphological operations, and connected components without relying on high-level OpenCV wrappers for the logic.

## Features

- **Custom PNG Encoder:** Implemented manual PNG chunk packing (IHDR, IDAT, CRC) using `struct` and `zlib`.
- **Statistical Background Modeling:** Builds a model using temporal Mean and Variance of initial frames.
- **Mahalanobis Distance Masking:** Uses statistical distance to identify foreground pixels, providing better robustness than simple subtraction.
- **Manual Morphological Ops:** From-scratch implementation of `Erosion` and `Dilation` logic.
- **BFS-based Connected Components:** Custom implementation of 4/8-connectivity labeling to filter out noise based on area.
- **Alpha-Blended Object Removal:** Seamlessly removes detected motion by blending background model data back into the sequence.

## Project Structure

```text
motion-detection/
├── src/
│   ├── main.py           # Pipeline orchestration & Alpha blending
│   ├── background.py     # Mean, Variance, and Mahalanobis logic
│   ├── morphology.py     # Manual Erode/Dilate implementations
│   ├── components.py     # BFS-based connected components labeling
│   └── io_utils.py       # Custom PNG writer and video utilities
├── requirements.txt      # Project dependencies
└── README.md

```

## 📊 The Pipeline

1. **Background Modeling:** The first  frames are used to calculate the pixel-wise mean () and variance ().
2. **Detection:** For new frames, the Mahalanobis distance is calculated:


3. **Cleaning:** Binary masks are cleaned using morphological opening (Erosion followed by Dilation).
4. **Filtering:** Connected component analysis removes small "salt and pepper" noise clusters.
5. **Removal:** The detected foreground is replaced with the background model using an alpha-blending transition.

## 🚦 Getting Started

### Prerequisites

* Python 3.8+
* NumPy, OpenCV (for video encoding), ImageIO

### Installation

```bash
git clone [https://github.com/aurangzeb200/motion-detection.git](https://github.com/aurangzeb200/motion-detection.git)
cd motion-detection
pip install -r requirements.txt

```

### Running the Detection

```bash
python -m main.py --input_folder ./data/sequence1 --output_folder ./results

```

## Results

The pipeline generates:

* `mean_frame.png` & `variance_frame.png` (The "memory" of the scene)
* Raw and cleaned binary masks for every frame.
* A final video showing the sequence with the moving subject removed.

## Motive

This project was developed for educational purposes to understand the "math under the hood." While libraries like OpenCV offer optimized versions of these functions (using C++), the algorithms here are implemented in Python/NumPy to demonstrate a deep understanding of the underlying theory.
