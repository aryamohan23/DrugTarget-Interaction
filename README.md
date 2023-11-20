# Scientific Machine Learning for Drug-Target Interaction Predictions
This project focuses on the accurate prediction of drug-target interactions (DTIs) using a scientific machine learning approach. It integrates physics-based knowledge into a data-driven model, enhancing prediction accuracy and model interpretability. Our model parameterizes equations derived from chemical interaction physics and employs a gated graph attention network to capture the intricacies of atomic interactions. This approach is vital for optimizing drug efficacy and guiding drug discovery efforts. Traditional methods of predicting DTIs either lack accuracy or are computationally expensive. Existing machine learning models often suffer from a lack of generalization due to the complex nature of molecular interactions. Our project addresses these challenges by integrating physics-based principles with advanced machine learning techniques to create a more accurate and interpretable model. Our model not only predicts the total binding affinity of drug compounds to target proteins but also calculates pairwise atom-atom energies, offering a detailed understanding of molecular interactions.

### Getting Started
#### Dependencies
- Python 3.6 or later
- PyTorch
- RDKit

#### To train the model:
python train.py

#### Files Description
utils.py: Utility functions for data processing and model evaluation.
train.py: Script to train the machine learning model.
models.py: Defines the gated graph attention network architecture.
data.py: Manages the loading and processing of drug-target interaction datasets.

### Acknowledgments
This work is adapted and inspired by the paper: https://pubs.rsc.org/en/content/articlelanding/2022/SC/D1SC06946B 
