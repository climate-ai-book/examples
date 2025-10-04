
# AI in Climate Science - Code Examples
Code examples and implementations from the book "AI in Climate Science: Machine Learning for Environmental Modeling and Prediction" by Prof. Sandeep Gupta, Prof. Budesh Kanwer, and Prof. Badrul Hisham Ahmad.
Published by Bentham Science Publishers, 2025.
About
This repository contains practical implementations of machine learning and deep learning techniques for climate science applications, organized by book chapters. Each chapter includes Jupyter notebooks with detailed explanations, Python scripts, and sample datasets.
Book Structure
Part I: Foundations and Methodologies

Chapter 1: Introduction to AI in Climate Science
Chapter 2: Climate Data Challenges and Preprocessing
Chapter 3: Machine Learning Fundamentals for Climate Applications

Part II: Atmospheric and Weather Systems

Chapter 4: AI-Enhanced Weather Prediction
Chapter 5: Extreme Weather Event Prediction
Chapter 6: Atmospheric Composition and Air Quality

Part III: Ocean, Cryosphere, and Terrestrial Systems

Chapter 7: Ocean Modeling with AI
Chapter 8: Cryosphere Monitoring and Prediction
Chapter 9: Vegetation and Ecosystem Dynamics
Chapter 10: Carbon Cycle and Greenhouse Gas Modeling

Part IV: Climate Impacts and Applications

Chapter 11: Renewable Energy and Climate Variability
Chapter 12: Agricultural and Food Security Applications
Chapter 13: Climate Risk Assessment and Adaptation
Chapter 14: Regional and Seasonal Climate Prediction
Chapter 15: Emerging Technologies and Future Directions

Installation
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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Repository Structure
examples/
├── README.md
├── requirements.txt
├── environment.yml
├── chapter-01-introduction/
│   ├── notebooks/
│   ├── scripts/
│   └── data/
├── chapter-02-preprocessing/
│   ├── notebooks/
│   ├── scripts/
│   └── data/
├── chapter-03-ml-fundamentals/
│   └── ...
└── ...
Usage
Running Jupyter Notebooks
bash# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
Navigate to the chapter folder and open the .ipynb files.
Running Scripts
bashcd chapter-XX-name/scripts
python example_script.py
Datasets
Sample datasets are provided in each chapter's data/ folder. For larger datasets, download instructions are provided in the respective chapter notebooks.
Common Data Sources Used:

ERA5 Reanalysis: ECMWF atmospheric data
MODIS: Satellite vegetation data
CMIP6: Climate model outputs
Custom datasets: Preprocessed for educational purposes

Requirements

Python 3.9 or higher
Key libraries: NumPy, Pandas, Scikit-learn, TensorFlow/PyTorch, Xarray
See requirements.txt for complete list

Interactive Notebooks
Try the examples online without installation:
Show Image
Documentation
Full documentation available at: https://climate-ai-book.readthedocs.io
Contributing
We welcome contributions! Please:

Fork the repository
Create a feature branch (git checkout -b feature/improvement)
Commit changes (git commit -am 'Add new example')
Push to branch (git push origin feature/improvement)
Create a Pull Request

Support

Issues: Report bugs or request features via GitHub Issues
Discussions: Join our Discussion Forum
Email: contact@climate-ai-book.org

Citation
If you use this code in your research, please cite:
bibtex@book{gupta2025aiclimate,
  title={AI in Climate Science: Machine Learning for Environmental Modeling and Prediction},
  author={Gupta, Sandeep and Kanwer, Budesh and Ahmad, Badrul Hisham},
  year={2025},
  publisher={Bentham Science Publishers}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
Authors

Prof. (Dr.) Sandeep Gupta - Poornima Institute of Engineering & Technology, India
Prof. (Dr.) Budesh Kanwer - Poornima Institute of Engineering & Technology, India
Prof. (Dr.) Badrul Hisham Ahmad - Universiti Teknikal Malaysia Melaka

Acknowledgments

Bentham Science Publishers
Contributors and reviewers
Climate science and ML communities

Website
Visit the book website: https://climate-ai-book.org

© 2025 AI in Climate Science. All rights reserved.
