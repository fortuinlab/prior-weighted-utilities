# UQ Evaluation With Prior-Weighted Utilities

This is the code for the ICML submission "Decision-Alignment & Prior-Weighted Utilities: Towards Meaningful Evaluation of Uncertainty Quantification".
In this README, we give instructions on how to run the experiments and refer to the paper for a detailed explanation of the setup.

## 🚀 Getting Started

Install `uv` and simply run
```
uv sync
```

## ⚙️ Running Experiments

### 📊 Benchmark Experiments

The code for the experiments on benchmark datasets can be found in `experiments/benchmark_datasets`.
There are two separate folders for the binary classification and regression experiments.
We now explain how to run the experiments for binary classification; regression follows analogously.
All required files can be found in the folder `experiments/benchmark_datasets/binary_classification`.

To fetch and preprocess the datasets, run `01_preprocess_datasets.py`.
A new folder appears at the root of the project containing the preprocessed data with all 100 different 5-fold splits.

To train the models for one specific dataset, repeat, and fold, run `02_train_models.py` with the respective arguments for `dataset`, `repeat`, and `fold`, for example, after navigating in the `binary_classification` folder, run
```
python 02_train_models.py --dataset bank --repeat 0 --fold 1
```
The results will be stored in the new folder `predictions`.
To train the models for several datasets over several repeats and folds, you can use the orchestration script `02_train_models.sh`.

For the evaluation for one dataset, run `03_evaluate_models.py` with the respective argument for `dataset`.
This can be orchestrated with `03_evaluate_models.sh`.
The results will be stored in the new folder `results`.

Running the sensitivity analysis for prior misspecification works just as regular evaluation, but using the Python script `04_sensitivity_analysis.py` (and the corresponding `.sh` file for orchestration).

### 🌳 Applied Case Studies

#### ⚡️ Electricity Market Bidding

The code for this case study can be found in `experiments/case_studies/electricity_market`.

To be able to preprocess the datasets, we first need to access the raw data.
The Belgian balancing price data is from Open Data Elia, license: CC BY 4.0.
Via [this link](https://opendata.elia.be/explore/dataset/ods134/information/?sort=-datetime&dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6Im9kczEzNCIsIm9wdGlvbnMiOnt9fSwiY2hhcnRzIjpbeyJhbGlnbk1vbnRoIjp0cnVlLCJ0eXBlIjoibGluZSIsImZ1bmMiOiJBVkciLCJ5QXhpcyI6ImFjZSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiNlNzU0MjAifV0sInhBeGlzIjoiZGF0ZXRpbWUiLCJtYXhwb2ludHMiOiIiLCJ0aW1lc2NhbGUiOiJ5ZWFyIiwic29ydCI6IiJ9XSwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D), the raw data can be downloaded.
The dataset needs to be stored as `balancing_prices.csv` in `data/case_studies/electricity_market/raw`.
The remaining two raw datasets are fetched automatically when running the script `01_preprocess_datasets.py`. However, for this to work, one needs an API key for the [ENTSO-E transparency platform](https://transparency.entsoe.eu).
One can generate a key in the account settings, after creating a (free) account.
Please store the key at the root of the project as `entsoe_key.txt`.
Now you can run `01_preprocess_datasets.py`.
The processed data will be stored in `data/case_studies/electricity_market/preprocessed`.

For model training and prediction run `02_train_models.py`.
Results will be stored in the `predictions` folder and model checkpoints for the GP in `models`.

For evaluation, run `03_evaluate_models.py`.
The results will be stored in `results`.

#### 🏦 Credit Approval

The code for this case study can be found in `experiments/case_studies/credit_approval`.

To be able to preprocess the datasets, we first need to access the raw data.
The Kaggle dataset can be downloaded [here](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview) and needs to be stored in `data/case_studies/credit_approval/raw/kaggle`.
The PAKDD dataset is fetched automatically when running the script `01_preprocess_datasets.py`.

For model training and prediction run `02_train_models.py` for a specific dataset (either `kaggle` or `pakdd`), repeat, and fold, or orchestrate the runs via `02_train_models.sh`.
Results will be stored in the `predictions` folder.

For evaluation, run `03_evaluate_models.py` for the `kaggle` or `pakdd` dataset separately.
The results will be stored in `results`.

#### 💶 P2P Lending

The code for this case study can be found in `experiments/case_studies/p2p_lending`.

To be able to preprocess the datasets, we first need to access the raw data.
The dataset can be downloaded [here](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and needs to be stored in `data/case_studies/p2p_lending/raw`.
We only need the dataset of _accepted_ loans.
Now you can run `01_preprocess_datasets.py`.
The processed data will be stored in `data/case_studies/p2p_lending/preprocessed`.

For model training and prediction run `02_train_models.py` for a repeat and fold, or orchestrate the runs via `02_train_models.sh`.
Results will be stored in the `predictions` folder.

For evaluation, run `03_evaluate_models.py`.
The results will be stored in `results`.

## 🌠 Visualizing Results

All scripts to create figures can be found in the folder `visualization`.
`case_study_figures.py` and `priors.py` can simply be executed and the resulting figure will be stored in a new folder `figures`.
The result boxplots can be created with the script `boxplots.py`, specifying the boxplot one wants to create: `benchmark`, `electricity`, or `credit_and_p2p`.
The sensitivity analysis results can be visualized with the script `sensitivity_analysis`, specifying the task (`binary_classification` or `regression`).