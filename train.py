import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import joblib

def main():
    print("=" * 60)
    print("Starting Machine Learning Pipeline - Kerala Flood Dataset")
    print("=" * 60)
    
    # 1. Load Dataset
    dataset_path = "dataset/flood.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_kerala_dataset.py first.")
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset Loaded: {dataset_path}")
    print(f"Dataset Shape: {df.shape}")
    
    # 2. Extract features and target
    # Independent features: Cloud Cover, ANNUAL, Jan-Feb, Mar-May, Jun-Sep
    features = ['Cloud Cover', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep']
    X = df[features]
    y = df['flood']
    
    # Print head of X and y
    print("\nTraining Features (X):")
    print(X.head())
    print("\nTarget Variable (y):")
    print(y.head())
    
    # 3. Train-Test Split (test_size=0.25, random_state=10, no stratification to match PDF)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 4. Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler as transform.save in the root directory
    scaler_path = "transform.save"
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler saved to: {scaler_path}")
    
    # 5. Model Training and Comparison
    print("\n--- Training Models ---")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=10),
        'Random Forest': RandomForestClassifier(random_state=10, n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=10),
        'XGBoost': XGBClassifier(random_state=10, eval_metric='logloss')
    }
    
    comparison_results = []
    trained_models = {}
    
    # Ensure static/images directory exists for saving plots
    os.makedirs(os.path.join("static", "images"), exist_ok=True)
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy Score: {acc * 100:.2f}%")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        
        comparison_results.append({
            'Model': name,
            'Accuracy': acc * 100,
            'Precision': prec,
            'Recall': rec
        })
        trained_models[name] = (model, acc)
        
    # Print Comparison Table
    print("\n" + "="*60)
    print("                      MODEL COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 60)
    for res in comparison_results:
        print(f"{res['Model']:<20} | {res['Accuracy']:.2f}%     | {res['Precision']:.2f}        | {res['Recall']:.2f}")
    print("="*60)
    
    # 6. Save the Best Model
    # Choose among XGBoost and Gradient Boosting as the selected final model if accuracies are similar,
    # or select the absolute highest accuracy model.
    # To match the PDF "XGBoost (Selected)" or "xgboost", we default to saving the best model, prioritizing XGBoost/Gradient Boosting.
    best_candidate_names = ['XGBoost', 'Gradient Boosting', 'Random Forest', 'Decision Tree', 'KNN']
    best_model_name = None
    best_accuracy = -1
    
    for name in best_candidate_names:
        if name in trained_models:
            _, acc = trained_models[name]
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                
    best_model, _ = trained_models[best_model_name]
    print(f"\n--> Selecting the Best Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")
    
    model_path = "floods.save"
    joblib.dump(best_model, model_path)
    print(f"Best Model saved as: {model_path}")
    print("\nPipeline execution completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
