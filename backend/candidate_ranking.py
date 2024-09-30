import argparse
import pandas as pd
import numpy as np
import sys
import json
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

app = Flask(__name__)

def load_and_prepare_data(file_path, is_job_roles=False):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Handle NaN values
    df = df.dropna().reset_index(drop=True)
    print(f"Missing values in {file_path}:\n", df.isnull().sum())

    if not is_job_roles:
        if 'skills' not in df.columns:
            if 'Skills' in df.columns:
                print("Warning: Renaming column 'Skills' to 'skills'")
                df.rename(columns={'Skills':'skills'}, inplace=True)
            else:
                raise ValueError("Neither 'skills' nor 'Skills' column found in the DataFrame.")

        df['skills'] = df['skills'].astype(str)

    print("\nColumn data types:")
    print(df.dtypes)

    return df

def engineer_features(df):
    df = df.copy()
    if 'experience' in df.columns:
        df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
        df['experience'] = df['experience'].apply(lambda x: min(x, 30) if pd.notnull(x) else x)

    def skill_score(x):
        return len(x.split(',')) if isinstance(x, str) else 0

    if 'skills' in df.columns:
        df['skill_match_score'] = df['skills'].apply(skill_score)

    return df

def extract_skills_from_training_data(df1, df2):
    skills1 = set(df1['skills'].str.split(',', expand=True).stack().str.strip().unique())
    skills2 = set(df2['Skills'].str.split(',', expand=True).stack().str.strip().unique())
    return skills1.union(skills2)

def train_models(df1, df2):
    # Combine the dataframes
    df1['source'] = 'main'
    df2['source'] = 'job_roles'
    df2 = df2.rename(columns={'Category': 'Job_Role_Skill_Category', 'Job Role': 'profile_title', 'Skills': 'skills'})
    df = pd.concat([df1, df2], ignore_index=True)

    df['target'] = 1  # Dummy target for training on skills only

    # Define the features (X) and target (y) for classification and regression
    X = df[['skills', 'experience', 'Job_Role_Skill_Category']]
    y_classification = df['target']
    y_regression = df['Net_Salary'] if 'Net_Salary' in df.columns else df['target']  # Use dummy target if Net_Salary is not available

    # Remove rows with NaN in target variables
    mask = ~y_classification.isna() & ~y_regression.isna()
    X = X[mask]
    y_classification = y_classification[mask]
    y_regression = y_regression[mask]

    # Split the data into training and test sets
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42)
    
    X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42)

    # Define the preprocessing steps for both numeric and categorical features
    categorical_features = ['skills', 'Job_Role_Skill_Category']
    numeric_features = ['experience']

    # Add SimpleImputer to the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Create pipelines for classification and regression
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    reg = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the models
    try:
        clf.fit(X_train, y_class_train)
        reg.fit(X_train_reg, y_reg_train)
        print("Models trained on combined data from both CSV files.")
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None  # Return None for both models if training fails

    return clf, reg

def predict_and_rank(class_model, reg_model, new_data, required_role, extracted_skills, skills_weight, top_n=5):
    def filter_candidates(df, role):
        return df[df['profile_title'].str.contains(role, case=False, na=False)]

    filtered_data = filter_candidates(new_data, required_role)

    # If we don't have enough candidates, gradually relax the filter
    if len(filtered_data) < top_n:
        role_words = required_role.split()
        for word in role_words:
            additional_candidates = filter_candidates(new_data, word)
            filtered_data = pd.concat([filtered_data, additional_candidates]).drop_duplicates()
            if len(filtered_data) >= top_n:
                break

    # If still not enough, use all data
    if len(filtered_data) < top_n:
        filtered_data = new_data

    filtered_data = filtered_data.reset_index(drop=True)

    filtered_data['skills'] = filtered_data['skills'].astype(str)
    filtered_data['skill_match_score'] = filtered_data['skills'].apply(
        lambda x: len(set(x.split(',')).intersection(extracted_skills))
    )

    filtered_data['Job_Role_Skill_Category'] = required_role

    if class_model is None or reg_model is None:
        print("Error: Models were not successfully trained. Cannot perform predictions.")
        return None

    X_new = filtered_data[['skills', 'experience', 'Job_Role_Skill_Category']]

    classification_predictions = class_model.predict(X_new)
    salary_predictions = reg_model.predict(X_new)

    max_experience = filtered_data['experience'].max()
    max_skill_match = filtered_data['skill_match_score'].max()

    normalized_experience = filtered_data['experience'] / max_experience if max_experience > 0 else 0
    normalized_skill_match = filtered_data['skill_match_score'] / max_skill_match if max_skill_match > 0 else 0

    experience_weight = 1 - skills_weight

    weighted_scores = (
        experience_weight * normalized_experience +
        skills_weight * normalized_skill_match
    ) * 100  # Scale to 0-100 range

    weighted_scores = weighted_scores * classification_predictions

    filtered_data['Weighted_Score'] = weighted_scores
    filtered_data['Predicted_Salary'] = salary_predictions

    # Ensure we return exactly top_n candidates, or all if less than top_n are available
    top_n = min(top_n, len(filtered_data))
    top_candidates = filtered_data.nlargest(top_n, 'Weighted_Score')

    top_candidates['Classification_Prediction'] = classification_predictions[top_candidates.index]

    return top_candidates[['candidate_id', 'full_name', 'experience', 'skills', 'profile_title', 'Classification_Prediction', 'Weighted_Score', 'Predicted_Salary']]

@app.route('/rank-candidates', methods=['POST'])

def rank_candidates():
    data=request.json
    required_role = data['requiredRole']
    top_n = data['topN']
    skills_weight = data['skillsWeight']
    main_data_file_path = data['mainDataFilePath']
    job_roles_data_file_path = data['jobRolesDataFilePath']
    test_data_file_path = data['testDataFilePath']

    print(f"Processing request for role: {required_role}, top {top_n} candidates, skills weight: {skills_weight}")

    main_df = load_and_prepare_data(main_data_file_path)
    job_roles_df = load_and_prepare_data(job_roles_data_file_path, is_job_roles=True)

    main_df = engineer_features(main_df)
    job_roles_df = engineer_features(job_roles_df)

    extracted_skills = extract_skills_from_training_data(main_df, job_roles_df)

    trained_class_model, trained_reg_model = train_models(main_df, job_roles_df)

    if trained_class_model is None or trained_reg_model is None:
        return jsonify({"error": "Failed to train models"}), 500

    new_df = load_and_prepare_data(test_data_file_path)
    new_df = engineer_features(new_df)

    results = predict_and_rank(trained_class_model, trained_reg_model, new_df, required_role, extracted_skills, skills_weight, top_n)

    if results is not None and not results.empty:
        results_dict = results.to_dict(orient='records')
        return jsonify(results_dict)
    else:
        return jsonify({"error": "No results found or an error occurred"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)