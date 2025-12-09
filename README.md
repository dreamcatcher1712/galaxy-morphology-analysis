Galaxy Morphology Analysis with Machine Learning

Automated galaxy morphology classification and anomaly detection using classical ML, CNNs, and unsupervised learning on the Galaxy Zoo 2 dataset.

📋 Project Overview
This project implements a comprehensive machine learning pipeline to:

Classify 138,693 galaxies into 3 morphological types (Smooth, Disk, Spiral)
Detect 445 rare/anomalous galaxies using consensus anomaly detection
Compare classical ML models vs. shallow CNN performance
Explore results through an interactive Streamlit dashboard

Best Results: SVM achieved 94.85% F1-score with 99.97% AUROC
Key Features

- Classical ML Models: Logistic Regression, Random Forest, SVM with class imbalance handling
- Shallow CNN: 4-layer convolutional network with data augmentation
- Unsupervised Learning: K-Means, DBSCAN clustering + 3 anomaly detection methods
- SQL Integration: Queryable SQLite database with 138k+ galaxies
- Interactive Dashboard: Streamlit app for result exploration
- Reproducible: Complete pipeline with documented code

Consensus Anomalies: 445 galaxies (0.32%) flagged by all 3 methods
Methods: Isolation Forest, Local Outlier Factor, PCA Reconstruction Error

Quick Start
Prerequisites
bashPython 3.8+
8GB RAM minimum
GPU optional (for CNN training)
Installation

Clone the repository

bashgit clone https://github.com/yourusername/galaxy-morphology-analysis.git
cd galaxy-morphology-analysis

Install dependencies

bashpip install -r requirements.txt

Download the dataset

bashpython download_data.py
This will download the Galaxy Zoo 2 dataset from Hugging Face (~2GB).
📁 Project Structure
galaxy-morphology-analysis/
│
├── requirements.txt                   # Python dependencies
│
├── 1_eda.py                          # Exploratory Data Analysis
├── 2_ml_models.py                    # Classical ML training
├── 3_cnn.py                          # CNN training
├── 4_unsupervised.py                 # Clustering & anomaly detection
├── 5_sql.py                          # SQL database integration
│
├── app.py                            # Streamlit dashboard
├── image_loader.py                   # Helper functions
├── sql_utils.py                      # Database utilities
│
├── outputs/                          # Generated results
│   ├── eda/                          # EDA visualizations
│   ├── models/                       # Trained model files (.pkl)
│   ├── results/                      # Performance metrics & plots
│   ├── cnn/                          # CNN results & weights
│   ├── unsupervised/                 # Clustering & anomaly results
│   └── sql/                          # SQL database & queries
│
├── gz2_prepared/                     # Dataset (after download)
│
├── report/                           # IEEE conference paper
│   ├── conference_paper.pdf
│
│

Key Findings
1. Classical ML vs CNN

SVM outperformed shallow CNN by 28.19 percentage points (F1-score)
Classical models more effective with engineered features vs. raw images
CNNs require deeper architectures for competitive performance

2. Class Imbalance Handling

Random undersampling + balanced class weights most effective
Reduced majority class from 97k → 70k samples
Improved minority class (Disk) detection significantly

3. Feature Importance
Top 5 discriminative features:

spiral_fraction (20.2%)
has_spiral_arms_yes_fraction (16.8%)
smooth_or_featured_disk_fraction (12.5%)
disk_fraction (11.7%)
has_spiral_arms_no_fraction (10.1%)

4. Anomaly Detection

445 consensus anomalies represent truly unusual objects
12-18% overlap with human-labeled "odd" galaxies
Methods capture different aspects of anomalousness

SQL Database Schema
The SQLite database contains 5 normalized tables:
sql-- Galaxy metadata
CREATE TABLE galaxies (
    galaxy_id INTEGER PRIMARY KEY,
    smooth_fraction REAL,
    disk_fraction REAL,
    spiral_fraction REAL,
    galaxy_type INTEGER,
    galaxy_type_name TEXT
);

-- ML predictions
CREATE TABLE ml_predictions (
    galaxy_id INTEGER,
    model_name TEXT,
    predicted_class INTEGER,
    confidence REAL
);

-- Clustering results
CREATE TABLE clustering_results (
    galaxy_id INTEGER PRIMARY KEY,
    kmeans_k3_cluster INTEGER,
    dbscan_cluster INTEGER,
    tsne_1 REAL,
    tsne_2 REAL
);

-- Anomaly scores
CREATE TABLE anomaly_scores (
    galaxy_id INTEGER PRIMARY KEY,
    isolation_forest_anomaly INTEGER,
    lof_anomaly INTEGER,
    pca_anomaly INTEGER,
    consensus_anomaly INTEGER
);

-- Model performance
CREATE TABLE model_performance (
    model_name TEXT,
    accuracy REAL,
    f1_macro REAL,
    auroc REAL
);
Example Queries
Find top 10 anomalous spiral galaxies:
sqlSELECT g.galaxy_id, g.spiral_fraction, a.pca_reconstruction_error
FROM galaxies g
JOIN anomaly_scores a ON g.galaxy_id = a.galaxy_id
WHERE g.galaxy_type_name = 'Spiral' AND a.consensus_anomaly = 1
ORDER BY a.pca_reconstruction_error DESC
LIMIT 10;

Technologies Used

Data Processing: Pandas, NumPy, HuggingFace Datasets
Machine Learning: Scikit-learn, Imbalanced-learn
Deep Learning: PyTorch, Torchvision
Visualization: Matplotlib, Seaborn, Plotly
Database: SQLite3
Dashboard: Streamlit
Dimensionality Reduction: PCA, t-SNE

Dataset
Source: Galaxy Zoo 2 Dataset (Hugging Face)
Statistics:

Size: 138,693 galaxies
Image Format: 128×128 RGB
Features: 76 columns (38 fraction features, 38 binary labels)
Classes: Smooth (70.1%), Disk (4.8%), Spiral (24.6%)

License
This project is licensed under the MIT License - see the LICENSE file for details.
Author
Ayushi Gupta

Acknowledgments

Galaxy Zoo Team - For crowdsourced labels and dataset
Hugging Face - For dataset hosting and tools
SDSS Collaboration - For galaxy imaging data
