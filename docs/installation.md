# Installation Guide
This guide will help you set up the environment needed to run the code examples from "AI in Climate Science".
Prerequisites

Python 3.9 or higher
Git (for cloning the repository)
8GB+ RAM recommended for running examples
GPU optional but recommended for deep learning examples

Quick Install
Option 1: Using Conda (Recommended)
bash# Clone the repository
git clone https://github.com/climate-ai-book/examples.git
cd examples

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate climate-ai
Option 2: Using pip
bash# Clone the repository
git clone https://github.com/climate-ai-book/examples.git
cd examples

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Verify Installation
Test your installation by running:
pythonimport numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import sklearn

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Xarray version:", xr.__version__)
print("TensorFlow version:", tf.__version__)
print("Scikit-learn version:", sklearn.__version__)
Core Dependencies
Scientific Computing

numpy - Numerical computing
pandas - Data manipulation
scipy - Scientific computing
matplotlib - Plotting
seaborn - Statistical visualization

Machine Learning

scikit-learn - Classical ML algorithms
tensorflow or pytorch - Deep learning
xgboost - Gradient boosting
lightgbm - Light gradient boosting

Climate-Specific Libraries

xarray - Multi-dimensional labeled arrays
netCDF4 - NetCDF file format support
cftime - Climate calendar handling
cartopy - Cartographic projections

Jupyter Environment

jupyterlab - Interactive notebooks
ipywidgets - Interactive widgets
notebook - Jupyter notebook

GPU Support (Optional)
For GPU acceleration with TensorFlow:
bash# Install TensorFlow with GPU support
pip install tensorflow-gpu

# Verify GPU availability
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
For PyTorch with GPU:
bash# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
Common Issues
Issue: Import errors for climate libraries
Solution: Install climate-specific packages via conda-forge:
bashconda install -c conda-forge xarray netcdf4 cartopy
Issue: TensorFlow/PyTorch compatibility
Solution: Use only one deep learning framework at a time. Comment out the unused one in requirements.txt.
Issue: Memory errors with large datasets
Solution: Use Dask for out-of-core computation:
bashpip install dask[complete]
Dataset Downloads
Some examples require downloading climate datasets. See the Datasets page for instructions.
Common data sources:

ERA5: Copernicus Climate Data Store
MODIS: NASA Earth Data
CMIP6: ESGF

Running Examples
Jupyter Notebooks
bash# Start Jupyter Lab
jupyter lab

# Navigate to chapter folders and open .ipynb files
Python Scripts
bash# Navigate to chapter folder
cd chapter-01-introduction/scripts

# Run script
python pinn_implementations.py
Updating the Environment
To update all packages:
bash# With conda
conda update --all

# With pip
pip install --upgrade -r requirements.txt



