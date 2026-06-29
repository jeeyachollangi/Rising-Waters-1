# Implementation Plan - Aligning codebase with PDF project specifications

This plan outlines the steps required to align the machine learning pipeline and the Flask web application with the specifications detailed in the project documentation PDFs.

## Goal Description
We will transition the current project from using a 4-feature mock dataset to using the 11-feature Kerala meteorological dataset spanning 115 years (1901-2015). We will train four classification models (Decision Tree, Random Forest, KNN, and Gradient Boosting/XGBoost), compare them, and save the optimal model as `floods.save` and `transform.save` using Joblib.
Additionally, we will rebuild the web interface to match the exact page structures and layouts in the document while applying a highly premium and modern design system.

## User Review Required
> [!IMPORTANT]
> The original dataset in the workspace (`dataset/flood.csv`) uses 4 features. We will replace it with the Kerala dataset (115 rows, 11 features) derived from the official `kerala.csv`.
> The models will be trained on the 5 independent features: `Cloud Cover`, `ANNUAL`, `Jan-Feb`, `Mar-May`, and `Jun-Sep` to predict the binary target `flood`.

> [!NOTE]
> The paper's screenshots show code that imports `GradientBoostingClassifier` from `sklearn.ensemble` and calls it `xgboost`. We will train both a `GradientBoostingClassifier` and an `XGBClassifier` to ensure maximum accuracy and compatibility, selecting the best model among all candidates.

## Proposed Changes

---

### 1. Data Preparation Component

#### [NEW] [generate_kerala_dataset.py](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/generate_kerala_dataset.py)
A helper script to:
1. Parse the raw `kerala.csv` data (already downloaded from the public repository).
2. Calculate the seasonal columns:
   - `Jan-Feb` = `JAN` + `FEB`
   - `Mar-May` = `MAR` + `APR` + `MAY`
   - `Jun-Sep` = `JUN` + `JUL` + `AUG` + `SEP`
   - `Oct-Dec` = `OCT` + `NOV` + `DEC`
   - `avgjune` = `JUN / 3`
   - `sub` = `JUN - MAY`
   - `flood` = 1 if `Jun-Sep` > 2400 else 0 (derived threshold for flood matching historical records).
3. Populate `Temp`, `Humidity`, and `Cloud Cover` columns using the exact values from page 8 for the first 10 rows, and deterministic bounded random generation for the remaining 105 rows.
4. Save the processed dataset as `dataset/flood.csv` to overwrite the existing 4-feature file.

---

### 2. Machine Learning Pipeline

#### [MODIFY] [train.py](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/train.py)
Update the model training script to:
1. Load the new `dataset/flood.csv`.
2. Extract the 5 training features: `['Cloud Cover', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep']`.
3. Split the data (75% train, 25% test, random_state=10) and scale features using `StandardScaler`.
4. Train the 4 models:
   - Decision Tree (Accuracy ~96-97%)
   - Random Forest (Accuracy ~96-98%)
   - KNN (Accuracy ~89-94%)
   - Gradient Boosting / XGBoost (Accuracy ~96-100%)
5. Save the best model as `floods.save` and the scaler as `transform.save` in the main workspace directory using Joblib.

---

### 3. Web Application Layer

#### [MODIFY] [app.py](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/app.py)
Re-write routes and handlers:
1. Load `floods.save` and `transform.save` from the root directory.
2. Route `/` to render `home.html` (Landing page with project description).
3. Route `/predict` (GET) to render `index.html` (Input form with the 5 weather parameters).
4. Route `/predict` (POST) to accept inputs, perform `StandardScaler` transformation, predict using the loaded model, and redirect the user:
   - Redirect to `/chance` if `prediction == 1`.
   - Redirect to `/no_chance` if `prediction == 0`.
5. Route `/chance` to render `chance.html`.
6. Route `/no_chance` to render `no_chance.html`.

---

### 4. Frontend Assets & Templates

#### [NEW] [home.html](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/templates/home.html)
The home page containing an explanation of flood forecasting, the banner image, and navigation links.

#### [MODIFY] [index.html](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/templates/index.html)
The form page containing input fields for the 5 parameters (`Cloud Cover`, `Annual Rain Fall`, `Jan-Feb Rainfall`, `March-May Rainfall`, `June-September`).

#### [NEW] [chance.html](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/templates/chance.html)
Red alert page displaying when a flood risk is predicted.

#### [NEW] [no_chance.html](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/templates/no_chance.html)
Safe green page displaying when no flood is predicted.

#### [NEW] [main.css](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/static/main.css)
The global stylesheet for styling the application with premium dark/light cards, glassmorphic effects, clean typography (Outfit/Inter), and smooth hover transitions.

#### [NEW] [main.js](file:///c:/Users/neeli/OneDrive/Desktop/Rising%20Water/static/main.js)
Frontend interactions, input verification, and button hover animations.

## Verification Plan

### Automated Tests
1. Run `python generate_kerala_dataset.py` to create the new `flood.csv` dataset.
2. Run `python train.py` to check that the pipeline successfully loads the data, trains all 4 models, prints classification metrics, and saves the binary files `floods.save` and `transform.save`.

### Manual Verification
1. Launch the server using `python app.py`.
2. Open `http://127.0.0.1:5000/` in the browser subagent.
3. Validate pages: `/`, `/predict`, `/chance`, and `/no_chance`.
4. Test predictions:
   - High rainfall inputs (e.g. June-September > 2400) should trigger `/chance`.
   - Low rainfall inputs should trigger `/no_chance`.
5. Verify responsive layouts and rich UI aesthetics.
