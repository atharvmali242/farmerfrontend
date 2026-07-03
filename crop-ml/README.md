# Crop Recommendation — ML Project

This is the machine learning version of the crop advisor: instead of fixed
rules, a model is trained on soil and climate data, and a Flask API serves
its predictions to a webpage.

## What's in this folder

```
crop-ml/
├── generate_dataset.py      # builds the training dataset
├── train_model.py           # trains the model on that dataset
├── data/
│   └── Crop_recommendation.csv
├── model/
│   ├── crop_model.pkl       # trained Random Forest model
│   ├── scaler.pkl           # feature scaler
│   └── label_encoder.pkl    # converts crop names <-> numbers
├── backend/
│   └── app.py                # Flask API that loads the model
└── frontend_ml.html          # webpage that calls the API
```

## About the dataset

The standard "Crop Recommendation Dataset" used in most student projects
(2200 rows, 22 crops, 7 features: N, P, K, temperature, humidity, ph,
rainfall) is hosted on Kaggle. I couldn't download it directly in this
environment, so `generate_dataset.py` builds a dataset with the **exact
same columns and the same 22 crops**, sampling values from realistic
agronomic ranges for each crop.

**Be upfront about this if your sir asks**: this is a synthetic dataset
built to match the real one's structure, not the original Kaggle file. If
he wants the real dataset, download `Crop_recommendation.csv` from
Kaggle yourself and drop it into `data/` with the same filename —
`train_model.py` will work on it unchanged, since the column names match.

Because this generated data has clean, non-overlapping ranges per crop,
the model scores 100% test accuracy — don't be surprised if it's lower
(more like 97–99%) on the real Kaggle data, where readings between
similar crops overlap more. That's normal and expected.

## How to run it, step by step

**1. Install the libraries** (only needed once):
```
pip install scikit-learn pandas numpy flask --break-system-packages
```

**2. Generate the dataset:**
```
cd crop-ml
python3 generate_dataset.py
```
This creates `data/Crop_recommendation.csv`.

**3. Train the model:**
```
python3 train_model.py
```
This prints accuracy and a report, and saves three files into `model/`.

**4. Start the backend API:**
```
python3 backend/app.py
```
Leave this running — it prints `Running on http://127.0.0.1:5000`.

**5. Open the frontend:**
Open `frontend_ml.html` directly in a browser (double-click it). Fill in
the seven values and click **Predict crop**. It calls the Flask server
running on your machine and shows the top 3 matches with confidence bars.

## How the model works, in plain terms

- **Random Forest classifier**: builds many small decision trees, each
  voting on the crop, then takes the majority answer. Chosen because it
  handles the kind of "if N is high AND rainfall is low" interactions
  common in this data, without much manual tuning.
- **Feature scaling**: rainfall is measured in hundreds of millimetres
  while pH is on a 0–14 scale — scaling puts every feature on the same
  footing so the model doesn't overweight whichever has bigger raw numbers.
- **Train/test split (80/20)**: the model only learns from 80% of the
  rows; the remaining 20% it has never seen are used to check accuracy
  honestly.

## API reference

**POST /predict**
```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.8, "humidity": 82,
  "ph": 6.5, "rainfall": 202.9
}
```
Returns:
```json
{
  "recommended_crop": "rice",
  "top_3": [
    {"crop": "rice", "confidence": 93.27},
    {"crop": "jute", "confidence": 6.73},
    {"crop": "pomegranate", "confidence": 0.0}
  ]
}
```

## Difference from the other crop-advisor.html

The earlier `crop-advisor.html` you have is a simple rule-based version —
no model, no backend, just plain-language questions matched against a
small hand-written table of crops. This ML version is the "real" data
science version: a model trained on numeric soil/climate readings. Good
to mention both if your project needs to show progression from a basic
version to a trained-model version.
