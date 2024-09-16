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

def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(file_path)

    df = df.dropna().reset_index(drop=True)

    print("Missing values:\n", df.isnull().sum())

    if 'skills' not in df.columns:
        if 'Skills' in df.columns:
            print("Warning: Renaming column 'Skills' to 'skills'")
            df.rename(columns={'Skills': 'skills'}, inplace=True)
        else:
            raise ValueError("Neither 'skills' nor 'Skills' column found in the DataFrame.")

    df['skills'] = df['skills'].astype(str)

    print("\nColumn data types:")
    print(df.dtypes)

    return df

def engineer_features(df):
    df = df.copy()
    df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
    df['experience'] = df['experience'].apply(lambda x: min(x, 30) if pd.notnull(x) else x)

    def skill_score(x):
        return len(x.split(',')) if isinstance(x, str) else 0

    df['skill_match_score'] = df['skills'].apply(skill_score)

    return df

def extract_skills_from_training_data(df):
    all_skills = df['skills'].str.split(',', expand=True).stack().str.strip().unique()
    return set(all_skills)

def train_models(df):
    X = df[['skills', 'experience', 'Job_Role_Skill_Category']]
    df['target'] = 1  # Dummy target for training on skills only
    y_classification = df['target']
    y_regression = df['Net_Salary']

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_classification, y_regression, test_size=0.2, random_state=42)

    categorical_features = ['skills', 'Job_Role_Skill_Category']
    numeric_features = ['experience']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    reg = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    clf.fit(X_train, y_class_train)
    reg.fit(X_train, y_reg_train)

    print("Models trained on skills, experience, and job role category.")

    return clf, reg

def predict_and_rank(class_model, reg_model, new_data, required_role, extracted_skills, skills_weight, top_n=5):
    def filter_candidates(df, role):
        return df[df['profile_title'].str.contains(role, case=False, na=False)]

    filtered_data = filter_candidates(new_data, required_role)

    # If we don't have enough candidates, gradually relax the filter
    if len(filtered_data) < top_n:
        # Split the role into words and try matching any of them
        role_words = required_role.split()
        for word in role_words:
            additional_candidates = filter_candidates(new_data, word)
            filtered_data = pd.concat([filtered_data, additional_candidates]).drop_duplicates()
            if len(filtered_data) >= top_n:
                break

    # If we still don't have enough, include all candidates
    if len(filtered_data) < top_n:
        filtered_data = new_data

    filtered_data = filtered_data.reset_index(drop=True)

    filtered_data['skills'] = filtered_data['skills'].astype(str)
    filtered_data['skill_match_score'] = filtered_data['skills'].apply(
        lambda x: len(set(x.split(',')).intersection(extracted_skills))
    )

    # Assign a default job role category for prediction
    filtered_data['Job_Role_Skill_Category'] = required_role

    X_new = filtered_data[['skills', 'experience', 'Job_Role_Skill_Category']]

    classification_predictions = class_model.predict(X_new)
    salary_predictions = reg_model.predict(X_new)

    # Normalize experience and skill_match_score
    max_experience = filtered_data['experience'].max()
    max_skill_match = filtered_data['skill_match_score'].max()

    normalized_experience = filtered_data['experience'] / max_experience if max_experience > 0 else 0
    normalized_skill_match = filtered_data['skill_match_score'] / max_skill_match if max_skill_match > 0 else 0

    experience_weight = 1 - skills_weight

    weighted_scores = (
        experience_weight * normalized_experience +
        skills_weight * normalized_skill_match
    ) * 100  # Scale to 0-100 range

    # Adjust scores based on model predictions
    weighted_scores = weighted_scores * classification_predictions

    # Add Weighted_Score and Predicted_Salary to the DataFrame
    filtered_data['Weighted_Score'] = weighted_scores
    filtered_data['Predicted_Salary'] = salary_predictions

    top_n = min(top_n, len(filtered_data))

    # Use nlargest to get the top candidates
    top_candidates = filtered_data.nlargest(top_n, 'Weighted_Score')

    top_candidates['Classification_Prediction'] = classification_predictions[top_candidates.index]

    return top_candidates[['candidate_id', 'full_name', 'experience', 'skills', 'profile_title', 'Classification_Prediction', 'Weighted_Score', 'Predicted_Salary']]

if __name__ == "__main__":
    try:
        if len(sys.argv) != 5:
            raise ValueError("Incorrect number of arguments. Usage: python candidate_ranking.py <required_role> <top_n> <skills_weight> <data_file_path>")

        required_role = sys.argv[1]
        top_n = int(sys.argv[2])
        skills_weight = float(sys.argv[3])
        data_file_path = sys.argv[4]

        if not 0 <= skills_weight <= 1:
            raise ValueError("Skills weight must be between 0 and 1")

        print(f"Processing request for role: {required_role}, top {top_n} candidates, skills weight: {skills_weight}")

        train_df = load_and_prepare_data('E:\\ats\\skills_and_salaries_categories_p.csv')
        train_df = engineer_features(train_df)
        extracted_skills = extract_skills_from_training_data(train_df)

        trained_class_model, trained_reg_model = train_models(train_df)

        new_df = load_and_prepare_data(data_file_path)
        new_df = engineer_features(new_df)

        results = predict_and_rank(trained_class_model, trained_reg_model, new_df, required_role, extracted_skills, skills_weight, top_n)

        print(results.to_json(orient='records'))
        sys.stdout.flush()

    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)