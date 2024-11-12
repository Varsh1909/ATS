# import pandas as pd
# import numpy as np
# import sys
# import json
# import traceback
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# def load_and_prepare_data(file_path):
#     try:
#         df = pd.read_csv(file_path, on_bad_lines='skip')
#     except TypeError:
#         df = pd.read_csv(file_path)

#     df = df.dropna().reset_index(drop=True)

#     print("Missing values:\n", df.isnull().sum())

#     if 'skills' not in df.columns:
#         if 'Skills' in df.columns:
#             print("Warning: Renaming column 'Skills' to 'skills'")
#             df.rename(columns={'Skills': 'skills'}, inplace=True)
#         else:
#             raise ValueError("Neither 'skills' nor 'Skills' column found in the DataFrame.")

#     df['skills'] = df['skills'].astype(str)

#     print("\nColumn data types:")
#     print(df.dtypes)

#     return df

# def engineer_features(df):
#     df = df.copy()
#     df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
#     df['experience'] = df['experience'].apply(lambda x: min(x, 30) if pd.notnull(x) else x)

#     def skill_score(x):
#         return len(x.split(',')) if isinstance(x, str) else 0

#     df['skill_match_score'] = df['skills'].apply(skill_score)

#     return df

# def extract_skills_from_training_data(df):
#     all_skills = df['skills'].str.split(',', expand=True).stack().str.strip().unique()
#     return set(all_skills)

# def train_models(df):
#     X = df[['skills', 'experience', 'Job_Role_Skill_Category']]
#     df['target'] = 1  # Dummy target for training on skills only
#     y_classification = df['target']
#     y_regression = df['Net_Salary']

#     X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
#         X, y_classification, y_regression, test_size=0.2, random_state=42)

#     categorical_features = ['skills', 'Job_Role_Skill_Category']
#     numeric_features = ['experience']

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])

#     clf = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
#     ])

#     reg = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
#     ])

#     clf.fit(X_train, y_class_train)
#     reg.fit(X_train, y_reg_train)

#     print("Models trained on skills, experience, and job role category.")

#     return clf, reg

# def predict_and_rank(class_model, reg_model, new_data, required_role, extracted_skills, skills_weight, top_n=5):
#     def filter_candidates(df, role):
#         return df[df['profile_title'].str.contains(role, case=False, na=False)]

#     filtered_data = filter_candidates(new_data, required_role)

#     # If we don't have enough candidates, gradually relax the filter
#     if len(filtered_data) < top_n:
#         # Split the role into words and try matching any of them
#         role_words = required_role.split()
#         for word in role_words:
#             additional_candidates = filter_candidates(new_data, word)
#             filtered_data = pd.concat([filtered_data, additional_candidates]).drop_duplicates()
#             if len(filtered_data) >= top_n:
#                 break

#     # If we still don't have enough, include all candidates
#     if len(filtered_data) < top_n:
#         filtered_data = new_data

#     filtered_data = filtered_data.reset_index(drop=True)

#     filtered_data['skills'] = filtered_data['skills'].astype(str)
#     filtered_data['skill_match_score'] = filtered_data['skills'].apply(
#         lambda x: len(set(x.split(',')).intersection(extracted_skills))
#     )

#     # Assign a default job role category for prediction
#     filtered_data['Job_Role_Skill_Category'] = required_role

#     X_new = filtered_data[['skills', 'experience', 'Job_Role_Skill_Category']]

#     classification_predictions = class_model.predict(X_new)
#     salary_predictions = reg_model.predict(X_new)

#     # Normalize experience and skill_match_score
#     max_experience = filtered_data['experience'].max()
#     max_skill_match = filtered_data['skill_match_score'].max()

#     normalized_experience = filtered_data['experience'] / max_experience if max_experience > 0 else 0
#     normalized_skill_match = filtered_data['skill_match_score'] / max_skill_match if max_skill_match > 0 else 0

#     experience_weight = 1 - skills_weight

#     weighted_scores = (
#         experience_weight * normalized_experience +
#         skills_weight * normalized_skill_match
#     ) * 100  # Scale to 0-100 range

#     # Adjust scores based on model predictions
#     weighted_scores = weighted_scores * classification_predictions

#     # Add Weighted_Score and Predicted_Salary to the DataFrame
#     filtered_data['Weighted_Score'] = weighted_scores
#     filtered_data['Predicted_Salary'] = salary_predictions

#     top_n = min(top_n, len(filtered_data))

#     # Use nlargest to get the top candidates
#     top_candidates = filtered_data.nlargest(top_n, 'Weighted_Score')

#     top_candidates['Classification_Prediction'] = classification_predictions[top_candidates.index]

#     return top_candidates[['candidate_id', 'full_name', 'experience', 'skills', 'profile_title', 'Classification_Prediction', 'Weighted_Score', 'Predicted_Salary']]

# if __name__ == "__main__":
#     try:
#         if len(sys.argv) != 5:
#             raise ValueError("Incorrect number of arguments. Usage: python candidate_ranking.py <required_role> <top_n> <skills_weight> <data_file_path>")

#         required_role = sys.argv[1]
#         top_n = int(sys.argv[2])
#         skills_weight = float(sys.argv[3])
#         data_file_path = sys.argv[4]

#         if not 0 <= skills_weight <= 1:
#             raise ValueError("Skills weight must be between 0 and 1")

#         print(f"Processing request for role: {required_role}, top {top_n} candidates, skills weight: {skills_weight}")

#         train_df = load_and_prepare_data('E:\\ats\\skills_and_salaries_categories_p.csv')
#         train_df = engineer_features(train_df)
#         extracted_skills = extract_skills_from_training_data(train_df)

#         trained_class_model, trained_reg_model = train_models(train_df)

#         new_df = load_and_prepare_data(data_file_path)
#         new_df = engineer_features(new_df)

#         results = predict_and_rank(trained_class_model, trained_reg_model, new_df, required_role, extracted_skills, skills_weight, top_n)

#         print(results.to_json(orient='records'))
#         sys.stdout.flush()

#     except Exception as e:
#         error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
#         print(json.dumps({"error": error_message}), file=sys.stderr)
#         sys.exit(1)

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
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

app = Flask(__name__)

def load_and_prepare_data(file_path, is_job_roles=False, parse_dates=False):
    try:
        if parse_dates:
            df = pd.read_csv(file_path, on_bad_lines='skip', parse_dates=['date'])
        else:
            df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        return None

    # Handle NaN values
    df = df.dropna().reset_index(drop=True)
    print(f"Missing values in {file_path}:\n", df.isnull().sum())

    if not is_job_roles:
        # Handle 'skills' column
        skills_column = next((col for col in df.columns if col.lower() == 'skills'), None)
        if skills_column:
            if skills_column != 'skills':
                print(f"Warning: Renaming column '{skills_column}' to 'skills'")
                df.rename(columns={skills_column: 'skills'}, inplace=True)
        else:
            raise ValueError("Neither 'skills' nor any case variation of 'skills' column found in the DataFrame.")

        df['skills'] = df['skills'].astype(str)
    else:
        # Handle 'Skills' column for job roles data
        skills_column = next((col for col in df.columns if col.lower() == 'skills'), None)
        if skills_column:
            if skills_column != 'skills':
                print(f"Info: Renaming column '{skills_column}' to 'skills' in job roles data")
                df.rename(columns={skills_column: 'skills'}, inplace=True)
        else:
            print("Error: 'skills' column not found in job roles data.")
            raise ValueError("'skills' column is required in the job roles data.")

    print(f"\nColumns in '{file_path}':")
    print(df.columns)
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
    skills2 = set(df2['skills'].str.split(',', expand=True).stack().str.strip().unique())
    return skills1.union(skills2)

def train_models(df1, df2):
    # Combine the dataframes
    df1['source'] = 'main'
    df2['source'] = 'job_roles'
    df2 = df2.rename(columns={
        'Category': 'Job_Role_Skill_Category',
        'Job Role': 'profile_title',
        'Skills': 'skills'  # This renaming is now handled in load_and_prepare_data
    })
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

    # Select and return the desired fields
    desired_fields = [
        'candidate_id', 'full_name', 'email', 'candidate_status', 'experience',
        'profile_title', 'source_name', 'date',
        'Classification_Prediction', 'Weighted_Score', 'Predicted_Salary'
    ]

    # Check if all desired fields exist
    missing_fields = [field for field in desired_fields if field not in top_candidates.columns]
    if missing_fields:
        print(f"Warning: The following fields are missing from the data and will not be returned: {missing_fields}")

    available_fields = [field for field in desired_fields if field in top_candidates.columns]

    return top_candidates[available_fields]

@app.route('/rank-candidates', methods=['POST'])
def rank_candidates():
    try:
        data = request.json

        required_role = data.get('requiredRole')
        top_n = int(data.get('topN', 5))
        skills_weight = float(data.get('skillsWeight', 0.5))
        main_data_file_path = data.get('mainDataFilePath')
        job_roles_data_file_path = data.get('jobRolesDataFilePath')
        test_data_file_path = data.get('testDataFilePath')
        start_date_str = data.get('startDate')
        end_date_str = data.get('endDate')

        print(f"Processing request for role: {required_role}, top {top_n} candidates, skills weight: {skills_weight}")
        print(f"Date filter: Start Date = {start_date_str}, End Date = {end_date_str}")

        # Load and prepare main and job roles data
        main_df = load_and_prepare_data(main_data_file_path)
        job_roles_df = load_and_prepare_data(job_roles_data_file_path, is_job_roles=True)

        if main_df is None or job_roles_df is None:
            return jsonify({"error": "Failed to load main or job roles data."}), 500

        main_df = engineer_features(main_df)
        job_roles_df = engineer_features(job_roles_df)

        extracted_skills = extract_skills_from_training_data(main_df, job_roles_df)

        trained_class_model, trained_reg_model = train_models(main_df, job_roles_df)

        if trained_class_model is None or trained_reg_model is None:
            return jsonify({"error": "Failed to train models"}), 500

        # Load and prepare test data
        new_df = load_and_prepare_data(test_data_file_path, parse_dates=True)
        if new_df is None:
            return jsonify({"error": "Failed to load test data."}), 500

        if 'date' not in new_df.columns:
            return jsonify({"error": "Test data must contain a 'date' column."}), 400

        # Initialize date filtering variables
        apply_date_filter = False
        if start_date_str and end_date_str:
            apply_date_filter = True
            try:
                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)
            except Exception as e:
                return jsonify({"error": f"Invalid date format: {e}"}), 400

            if start_date > end_date:
                return jsonify({"error": "'startDate' cannot be after 'endDate'."}), 400

        elif start_date_str or end_date_str:
            return jsonify({"error": "Both 'startDate' and 'endDate' must be provided if filtering by date."}), 400

        # Filter new_df based on the date range if applicable
        if apply_date_filter:
            new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
            before_filter_count = len(new_df)
            new_df = new_df[(new_df['date'] >= start_date) & (new_df['date'] <= end_date)]
            after_filter_count = len(new_df)
            print(f"Filtered test data from {before_filter_count} to {after_filter_count} records based on date.")

            if new_df.empty:
                return jsonify({"error": "No candidates found within the specified date range."}), 404
        else:
            print("No date filter applied.")

        new_df = engineer_features(new_df)

        results = predict_and_rank(
            trained_class_model, 
            trained_reg_model, 
            new_df, 
            required_role, 
            extracted_skills, 
            skills_weight, 
            top_n
        )

        if results is not None and not results.empty:
            # Convert 'date' to string format for JSON serialization
            if 'date' in results.columns:
                results['date'] = results['date'].dt.strftime('%Y-%m-%d')

            results_dict = results.to_dict(orient='records')
            return jsonify(results_dict)
        else:
            return jsonify({"error": "No results found or an error occurred"}), 404

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)