# AtomDetectorApp
AtomDetectorApp is a sophisticated image analysis tool for atomic-scale microscopy images, specifically designed for polarization vector analysis in atomic lattices. This application enables scientists to process microscopy images, detect atomic positions, and analyze crystal structures with high precision.

## Features

- **Image Processing**: Load and preprocess microscopy images with brightness equalization and intensity rescaling.
- **Atom Detection**: Automatically detect atoms in microscopy images using threshold-based detection with size filtering.
- **Manual Editing**: Add/remove detected regions and individual center points with intuitive interface.
- **Lattice Analysis**: Identify and analyze crystal lattice patterns from atomic coordinates.
- **Polarization Vectors**: Calculate and visualize polarization vectors between atomic positions.
- **Vector Customization**: Adjust vector scale for better visualization and toggle point display.
- **Data Export**: Save detected coordinates and analysis results to CSV files.

## Dependencies

- Python 3.7+
- NumPy
- OpenCV (cv2)
- pandas
- PyQt5
- SciPy
- scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AtomDetectorApp.git
cd AtomDetectorApp

# Install dependencies
pip install numpy opencv-python pandas pyqt5 scipy scikit-learn
```

## Usage

```bash
python atom_detector.py
```

### Basic Workflow

1. **Load an image**: Click "Load Image" to open a microscopy image.
2. **Preprocess the image**: Use brightness equalization and intensity rescaling if needed.
3. **Adjust thresholds**: Set min/max thresholds to detect atoms accurately.
4. **Fine-tune detection**: Modify atom size range parameters for precise detection.
5. **Manual editing**: Use drawing/erasing tools to refine detection.
6. **Calculate centers**: Find the center of each detected atom cluster.
7. **Save results**: Export atom coordinates to CSV for further analysis.

### Lattice Analysis

1. **Load coordinate files**: Import A and B atomic coordinates from CSV files.
2. **Calculate lattice centers**: Identify lattice centers based on A-atom positions.
3. **Compute polarization vectors**: Calculate displacement vectors from lattice centers to nearest B atoms.
4. **Adjust visualization**: Scale vectors and toggle point display for optimal visualization.

## Core Algorithm Design

### Atom Detection Algorithm

The application uses a multi-step approach for robust atom detection:

1. **Thresholding**: Apply adaptive thresholding to isolate atomic regions.
2. **Morphological Operations**: Use morphological transformations to connect and refine regions.
3. **Size Filtering**: Filter connected components based on size constraints to eliminate noise and artifacts.
4. **Manual Correction**: Allow user refinement to handle difficult cases.

### Lattice Analysis Algorithm

The lattice analysis identifies crystal structures using geometric relationships:

1. **Neighbor Search**: For each A atom, find its nearest neighbors using KD-Tree for efficient spatial queries.
2. **Corner Point Identification**: Identify the four corner points of each unit cell based on angle and distance relationships.
3. **Center Calculation**: Calculate the center of each identified unit cell.
4. **Clustering**: Apply DBSCAN clustering to merge and refine lattice centers, eliminating duplicates.

### Polarization Vector Calculation

Polarization vectors are calculated by:

1. **Nearest Neighbor Search**: For each lattice center, find the closest B atom.
2. **Vector Computation**: Calculate the displacement vector from center to B atom.
3. **Anomaly Filtering**: Filter out vectors with abnormal lengths (beyond 3 standard deviations).
4. **Visualization**: Render vectors with adjustable scale for clear visualization.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed for research in ferroelectric materials and crystallography
- Inspired by the need for better atomic-scale polarization analysis tools
