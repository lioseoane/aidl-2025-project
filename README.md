# Final Project AIDL

## Overview
This project is part of the final assignment for the AIDL postgraduate course of the UPC.

---

## Folder Structure
Below is the structure of the project directory along with a description of each folder and file:

```plaintext
project_name/
|
├── data/
│   ├── raw/             # Original raw data files (e.g., videos, images, sensor data)
│   ├── processed/       # Preprocessed data files (e.g., resized, normalized)
│   └── annotations/     # Ground truth or labeling files
|
├── notebooks/           # Jupyter notebooks for exploratory analysis, EDA, and experiments
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_visualization.ipynb
|
├── src/                 # Source code for the project
│   ├── __init__.py      # Makes src a package
│   ├── data/            # Scripts for data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── preprocess.py
│   ├── models/          # Model architectures and utilities
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── custom_model.py
│   ├── training/        # Training scripts and utilities
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── tracking/        # Human tracking-specific algorithms
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   └── postprocessing.py
│   └── utils/           # Helper functions and utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── metrics.py
|
├── scripts/             # Standalone scripts for specific tasks
│   ├── run_training.py  # Script to run training
│   ├── run_inference.py # Script for inference
│   └── run_evaluation.py# Script for evaluation
|
|
└── keypoint_model.pth   # Dummy saved model
|
|
├── requirements.txt     # Python dependencies
├── README.md            # Overview of the project
├── .gitignore           # Files and directories to ignore in git
└── LICENSE              # Licensing information
```

---

### Folder Descriptions

#### `data/`
- **raw/**: Contains the original, unmodified datasets.
- **processed/**: Includes preprocessed datasets (e.g., scaled, normalized).
- **annotations/**: Stores annotation files or labels for supervised learning tasks.

#### `notebooks/`
- Contains Jupyter notebooks for data analysis, visualization, and prototyping.

#### `src/`
- **data/**: Functions for loading and preprocessing data.
- **models/**: Neural network architectures and related utilities.
- **training/**: Scripts for model training and evaluation.
- **tracking/**: Algorithms and methods specific to human tracking.
- **utils/**: Miscellaneous utility functions, such as metrics and visualizations.

#### `scripts/`
- Standalone Python scripts for executing key tasks like training and inference.


---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/project_name.git
   cd project_name
   ```

2. Install dependencies:
   - Using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

3. Virtualenv created from Python version 3.11.11

---

## Usage
- Modify and run Jupyter notebooks in the `notebooks/` directory for exploration and prototyping.
- Use the `scripts/` directory for full pipeline execution.

---

## Contributing
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push the branch and create a pull request.

---

## License
See the `LICENSE` file for details.

