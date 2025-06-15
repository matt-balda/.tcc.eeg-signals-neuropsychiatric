# `BrainDx`: Binary and Multiclass Models for EEG Signals in Neuropsychiatric Disorder Classification

This project applies deep learning models to resting-state EEG data from Park et al. (2021).

[need to update]

[graphical abstract](https://www.elsevier.com/authors/tools-and-resources/visual-abstract).


### Directory organization

```
├── data/                # Directory containing all data files
│   ├── processed/       # Processed and cleaned data files
│   ├── raw/             # Original, immutable raw data files
│   └── synthetic/       # Synthetically generated data files
├── docs/                # Project documentation
│   ├── assets/          # Assets like images for the documentation
│   └── notebooks/       # Jupyter notebooks for exploration and analysis
├── imgs/                # Generated images, plots, and figures
├── models/              # Trained and serialized models
├── references/          # Reference data, lookup tables, or data dictionaries
├── results/             # Experiment results, predictions, and training logs
│   ├── binary/          # Results for binary classification models
│   ├── multiclass/      # Results for multiclass classification models
│   └── training_logs/   # Logs from model training sessions
├── src/                 # Source code for this project
│   ├── data/            # Scripts and modules for data processing and handling
│   ├── models/          # Scripts and modules for model training and evaluation
│   └── notebooks/       # Supporting code or modules used by the notebooks
├── .env                 # File for environment variables
├── .gitignore           # Specifies files and directories for Git to ignore
├── .python-version      # Specifies the project's Python version
├── app.py               # Main file for a web application (e.g., Streamlit, Flask)
├── init_pipeline.py     # Script to initialize a data or modeling pipeline
├── LICENSE              # Project's software license
├── mkdocs.yml           # Configuration file for the MkDocs documentation generator
├── pyproject.toml       # Poetry file for project metadata and dependencies
├── README.md            # General information about the project
├── requirements.txt     # List of Python dependencies for reproducibility
└── setup.py             # Setup script for making the project installable
```