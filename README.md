# Logistic Regression with Gradio and Flask UIs

Machine Learning (Lab) assignment, SZABIST Islamabad.
**Author:** Zain Ashfaq

This project trains a **logistic regression** classifier on two different classification datasets and serves each model through two web interfaces: **Gradio** and **Flask**.

## Datasets

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Name | Credit Card Default | Breast Biopsy |
| Source | ISLR `Default` (via Rdatasets) | MASS `biopsy` (via Rdatasets) |
| Rows | 10,000 | 699 |
| Target | `default` (No / Yes) | `class` (benign / malignant) |
| Features | `student`, `balance`, `income` | `V1` to `V9` (cell measurements) |
| Ignored columns | `rownames` | `rownames`, `ID` |

## Method

1. Load the CSV, remove duplicate rows and rows with a missing target.
2. Preprocess with a scikit-learn `Pipeline` and `ColumnTransformer`:
   - Numeric columns: median imputation, then standard scaling.
   - Categorical columns: most-frequent imputation, then one-hot encoding.
3. Split the data 80/20 (stratified, `random_state=42`).
4. Train `LogisticRegression` and evaluate on the test set.
5. Save each trained pipeline with `joblib` in `models/`.

## Results

| Dataset | Test accuracy |
|---|---|
| Credit Card Default | 97.30% |
| Breast Biopsy | 95.00% |

**Breast Biopsy:** the model made 7 mistakes out of 140 test rows (3 benign predicted as malignant, 4 malignant predicted as benign).

**Credit Card Default:** the classes are heavily imbalanced (only about 3% of customers default). The high accuracy is mostly because the model predicts "No" for almost everyone. For the "Yes" class, precision is 0.72 but recall is only 0.31, so most real defaulters are missed. Options for improving this include `class_weight="balanced"` or adjusting the decision threshold.

## Project structure

```
.
├── app_gradio.py       # Gradio UI (one tab per dataset)
├── app_flask.py        # Flask UI (one form per dataset)
├── train.py            # trains and evaluates both models
├── common.py           # shared loading and prediction code
├── config.py           # dataset paths, target columns, titles
├── download_data.py    # downloads the two CSV files into data/
├── requirements.txt
├── data/               # dataset1.csv, dataset2.csv
├── models/             # saved models (created by train.py)
└── templates/          # HTML templates for Flask
```

## How to run

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. download the datasets (skip if data/ already has the CSVs)
python download_data.py

# 4. train both models and print the evaluation reports
python train.py

# 5a. run the Gradio app  ->  http://127.0.0.1:7860
python app_gradio.py

# 5b. run the Flask app   ->  http://127.0.0.1:5000
python app_flask.py
```

Run the two apps one at a time.

## Using your own datasets

Edit `config.py` and set the CSV path, target column, title and any ID-like columns to drop for each dataset. Both UIs build their input fields automatically from the dataset columns, so no other code needs to change.

## Screenshots

Add your screenshots here, for example:

- Training output (`train.py`)
- Gradio app, Dataset 1 and Dataset 2
- Flask app, Dataset 1 and Dataset 2

## Tech stack

Python, pandas, scikit-learn, joblib, Gradio, Flask