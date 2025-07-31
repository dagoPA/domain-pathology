# Project: Domain Generalization for Computational Pathology

## 1. Overview

This project tackles the challenge of **domain generalization** in computational pathology. The goal is to develop robust tissue classification models that perform accurately on data from medical centers not seen during training.

The workflow is based on the **CAMELYON17** dataset, where each of the 5 participating medical centers is treated as a distinct **domain**. The core idea is to first extract high-level features from the Whole Slide Images (WSIs) and then use the **DomainBed** benchmark suite to train and evaluate various domain generalization algorithms.

---

## 2. Project Workflow

This project is structured to follow a clear pipeline from data acquisition to model evaluation, as outlined in the `notebooks/domain_pathology.ipynb` notebook.

### Step 1: Data Download & Preparation

The first step is to download the necessary data from the CAMELYON17 dataset.

* **Action**: Use the `source/dataset/download_dataset.py` script to fetch the WSIs. The notebook provides an example of how to download a small subset for development purposes.
* **Analysis**: The `source/analysis/summarize_dataset.py` script can be used to generate statistics and visualizations of the dataset, confirming the multi-domain (multi-center) nature of the data.

### Step 2: Feature Extraction `(TO-DO)`

This is a crucial step where meaningful representations are extracted from the raw WSIs. As noted in the project notebook, the implementation for this phase is pending.

* **Objective**: Process each WSI to generate a single feature vector that represents the entire slide. The plan is to use a dedicated slide-level feature extractor model (e.g., **TITAN**).
* **Status**: **This step needs to be implemented.** The notebook currently contains a placeholder for this logic. The implementation should iterate through the downloaded WSIs, pass them to the feature extractor, and save the resulting feature vectors for the next step.

### Step 3: Data Structuring for DomainBed `(TO-DO)`

Once the features are extracted, they must be organized in a format compatible with the DomainBed framework.

* **Objective**: Associate each feature vector with its corresponding label and, most importantly, its **domain ID** (i.e., the hospital it came from). This structured data should then be saved, for instance, in an HDF5 file or another suitable format.
* **Status**: **This step needs to be implemented.** The notebook describes the goal but does not contain the implementation code.

### Step 4: Training Models with DomainBed `(TO-DO)`

The final step is to use the prepared, domain-aware feature set to train and evaluate domain generalization models.

* **Objective**: Use DomainBed's command-line interface to train various algorithms (e.g., ERM, IRM, V-REx). The experiments should be set up to train on a subset of domains (hospitals) and test on a held-out, unseen domain to properly measure generalization performance.
* **Status**: **This step needs to be implemented.** The notebook shows a hypothetical command for running a DomainBed experiment, which will serve as a template for the actual training script execution.