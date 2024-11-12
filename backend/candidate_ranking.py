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

    df = df.dropna().reset_index(drop=True)
    print(f"Missing values in {file_path}:\n", df.isnull().sum())

    if not is_job_roles:
        skills_column = next((col for col in df.columns if col.lower() == 'skills'), None)
        if skills_column:
            if skills_column != 'skills':
                print(f"Warning: Renaming column '{skills_column}' to 'skills'")
                df.rename(columns={skills_column: 'skills'}, inplace=True)
        else:
            raise ValueError("'skills' column not found in the DataFrame.")
        df['skills'] = df['skills'].astype(str)
    else:
        skills_column = next((col for col in df.columns if col.lower() == 'skills'), None)
        if skills_column:
            if skills_column != 'skills':
                print(f"Info: Renaming column '{skills_column}' to 'skills' in job roles data")
                df.rename(columns={skills_column: 'skills'}, inplace=True)
        else:
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
    return df

def extract_skills_from_training_data(df1, df2):
    skills1 = set(df1['skills'].str.split(',', expand=True).stack().str.strip().unique())
    skills2 = set(df2['skills'].str.split(',', expand=True).stack().str.strip().unique())
    return skills1.union(skills2)

def train_models(df1, df2):
    df1['source'] = 'main'
    df2['source'] = 'job_roles'
    df2 = df2.rename(columns={
        'Category': 'Job_Role_Skill_Category',
        'Job Role': 'profile_title',
    })
    df = pd.concat([df1, df2], ignore_index=True)
    df['target'] = 1

    X = df[['skills', 'experience', 'Job_Role_Skill_Category']]
    y_classification = df['target']
    y_regression = df['Net_Salary'] if 'Net_Salary' in df.columns else df['target']

    mask = ~y_classification.isna() & ~y_regression.isna()
    X = X[mask]
    y_classification = y_classification[mask]
    y_regression = y_regression[mask]

    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42)

    categorical_features = ['skills', 'Job_Role_Skill_Category']
    numeric_features = ['experience']

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

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    reg = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    try:
        clf.fit(X_train, y_class_train)
        reg.fit(X_train_reg, y_reg_train)
        print("Models trained successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None

    return clf, reg

def predict_and_rank(class_model, reg_model, new_data, required_role, extracted_skills, top_n=5, max_experience=None, mandatory_skills=None, optional_skills=None):
    def filter_candidates(df, role):
        return df[df['profile_title'].str.contains(role, case=False, na=False)]

    filtered_data = filter_candidates(new_data, required_role)

    if len(filtered_data) < top_n:
        role_words = required_role.split()
        for word in role_words:
            additional_candidates = filter_candidates(new_data, word)
            filtered_data = pd.concat([filtered_data, additional_candidates]).drop_duplicates()
            if len(filtered_data) >= top_n:
                break

    if len(filtered_data) < top_n:
        filtered_data = new_data

    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data['skills'] = filtered_data['skills'].astype(str)

    # Apply experience filter
    if max_experience is not None:
        filtered_data = filtered_data[filtered_data['experience'] <= max_experience]

    # Check for mandatory and optional skills
    if mandatory_skills:
        filtered_data['has_mandatory_skills'] = filtered_data['skills'].apply(
            lambda x: all(skill.lower() in x.lower() for skill in mandatory_skills)
        )
        filtered_data = filtered_data[filtered_data['has_mandatory_skills']]
    else:
        filtered_data['has_mandatory_skills'] = True

    if optional_skills:
        filtered_data['has_optional_skills'] = filtered_data['skills'].apply(
            lambda x: any(skill.lower() in x.lower() for skill in optional_skills)
        )
    else:
        filtered_data['has_optional_skills'] = True

    if class_model is None or reg_model is None:
        print("Error: Models were not successfully trained. Cannot perform predictions.")
        return None

    # Add 'Job_Role_Skill_Category' if it doesn't exist
    if 'Job_Role_Skill_Category' not in filtered_data.columns:
        filtered_data['Job_Role_Skill_Category'] = required_role

    X_new = filtered_data[['skills', 'experience', 'Job_Role_Skill_Category']]
    classification_predictions = class_model.predict(X_new)
    salary_predictions = reg_model.predict(X_new)

    filtered_data['Classification_Prediction'] = classification_predictions
    filtered_data['Predicted_Salary'] = salary_predictions

    # Sort by Classification_Prediction and Predicted_Salary
    top_candidates = filtered_data.sort_values(['Classification_Prediction', 'Predicted_Salary'], ascending=[False, False]).head(top_n)

    desired_fields = [
        'candidate_id', 'full_name', 'email', 'experience',
        'profile_title', 'source_name', 'date', 'Classification_Prediction',
        'Predicted_Salary', 'skills', 'has_mandatory_skills', 'has_optional_skills'
    ]

    available_fields = [field for field in desired_fields if field in top_candidates.columns]
    
    # Extract top 5 skills for each candidate
    def extract_top_skills(skills_str, n=5):
        skills = skills_str.split(',')
        return ','.join(skills[:n])

    top_candidates['skills'] = top_candidates['skills'].apply(extract_top_skills)

    return top_candidates[available_fields]

@app.route('/rank-candidates', methods=['POST'])
def rank_candidates():
    try:
        data = request.json
        required_role = data.get('requiredRole')
        top_n = int(data.get('topN', 5))
        main_data_file_path = data.get('mainDataFilePath')
        job_roles_data_file_path = data.get('jobRolesDataFilePath')
        test_data_file_path = data.get('testDataFilePath')
        start_date_str = data.get('startDate')
        end_date_str = data.get('endDate')
        max_experience = data.get('maxExperience')
        mandatory_skills = data.get('mandatorySkills', [])
        optional_skills = data.get('optionalSkills', [])

        print(f"Processing request for role: {required_role}, top {top_n} candidates")
        print(f"Date filter: Start Date = {start_date_str}, End Date = {end_date_str}")
        print(f"Max Experience: {max_experience}")
        print(f"Mandatory Skills: {mandatory_skills}")
        print(f"Optional Skills: {optional_skills}")

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

        new_df = load_and_prepare_data(test_data_file_path, parse_dates=True)
        if new_df is None:
            return jsonify({"error": "Failed to load test data."}), 500

        if 'date' not in new_df.columns:
            return jsonify({"error": "Test data must contain a 'date' column."}), 400

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
            trained_class_model, trained_reg_model, new_df, required_role,
            extracted_skills, top_n, max_experience, mandatory_skills, optional_skills
        )

        if results is not None and not results.empty:
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

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from flask import Flask, request, jsonify
# import warnings
# import traceback

# warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

# app = Flask(__name__)

# def load_and_prepare_data(file_path, is_job_roles=False, parse_dates=False):
#     try:
#         if parse_dates:
#             df = pd.read_csv(file_path, on_bad_lines='skip', parse_dates=['date'])
#         else:
#             df = pd.read_csv(file_path, on_bad_lines='skip')
#     except Exception as e:
#         print(f"Error reading CSV file '{file_path}': {e}")
#         return None

#     df = df.dropna().reset_index(drop=True)
#     print(f"Missing values in {file_path}:\n", df.isnull().sum())

#     if not is_job_roles:
#         skills_column = next((col for col in df.columns if col.lower() == 'skills'), None)
#         if skills_column:
#             if skills_column != 'skills':
#                 print(f"Warning: Renaming column '{skills_column}' to 'skills'")
#                 df.rename(columns={skills_column: 'skills'}, inplace=True)
#         else:
#             raise ValueError("'skills' column not found in the DataFrame.")
#         df['skills'] = df['skills'].astype(str)
#     else:
#         skills_column = next((col for col in df.columns if col.lower() == 'skills'), None)
#         if skills_column:
#             if skills_column != 'skills':
#                 print(f"Info: Renaming column '{skills_column}' to 'skills' in job roles data")
#                 df.rename(columns={skills_column: 'skills'}, inplace=True)
#         else:
#             raise ValueError("'skills' column is required in the job roles data.")

#     print(f"\nColumns in '{file_path}':")
#     print(df.columns)
#     print("\nColumn data types:")
#     print(df.dtypes)
#     return df

# def engineer_features(df):
#     df = df.copy()
#     if 'experience' in df.columns:
#         df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
#         df['experience'] = df['experience'].apply(lambda x: min(x, 30) if pd.notnull(x) else x)
#     return df

# def extract_skills_from_training_data(df1, df2):
#     skills1 = set(df1['skills'].str.split(',', expand=True).stack().str.strip().unique())
#     skills2 = set(df2['skills'].str.split(',', expand=True).stack().str.strip().unique())
#     return skills1.union(skills2)

# def train_models(df1, df2):
#     df1['source'] = 'main'
#     df2['source'] = 'job_roles'
#     df2 = df2.rename(columns={
#         'Category': 'Job_Role_Skill_Category',
#         'Job Role': 'profile_title',
#     })
#     df = pd.concat([df1, df2], ignore_index=True)
#     df['target'] = 1

#     X = df[['skills', 'experience', 'Job_Role_Skill_Category']]
#     y_classification = df['target']
#     y_regression = df['Net_Salary'] if 'Net_Salary' in df.columns else df['target']

#     mask = ~y_classification.isna() & ~y_regression.isna()
#     X = X[mask]
#     y_classification = y_classification[mask]
#     y_regression = y_regression[mask]

#     X_train, X_test, y_class_train, y_class_test = train_test_split(
#         X, y_classification, test_size=0.2, random_state=42)
#     X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
#         X, y_regression, test_size=0.2, random_state=42)

#     categorical_features = ['skills', 'Job_Role_Skill_Category']
#     numeric_features = ['experience']

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline([
#                 ('imputer', SimpleImputer(strategy='mean')),
#                 ('scaler', StandardScaler())
#             ]), numeric_features),
#             ('cat', Pipeline([
#                 ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore'))
#             ]), categorical_features)
#         ])

#     clf = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
#     ])

#     reg = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
#     ])

#     try:
#         clf.fit(X_train, y_class_train)
#         reg.fit(X_train_reg, y_reg_train)
#         print("Models trained successfully.")
#     except Exception as e:
#         print(f"Error during model training: {e}")
#         return None, None

#     return clf, reg

# def predict_and_rank(class_model, reg_model, new_data, required_role, extracted_skills, top_n=5, max_experience=None, mandatory_skills=None, optional_skills=None):
#     def filter_candidates(df, role):
#         return df[df['profile_title'].str.contains(role, case=False, na=False)]

#     filtered_data = filter_candidates(new_data, required_role)

#     if len(filtered_data) < top_n:
#         role_words = required_role.split()
#         for word in role_words:
#             additional_candidates = filter_candidates(new_data, word)
#             filtered_data = pd.concat([filtered_data, additional_candidates]).drop_duplicates()
#             if len(filtered_data) >= top_n:
#                 break

#     if len(filtered_data) < top_n:
#         filtered_data = new_data

#     filtered_data = filtered_data.reset_index(drop=True)
#     filtered_data['skills'] = filtered_data['skills'].astype(str)

#     # Apply experience filter
#     if max_experience is not None:
#         filtered_data = filtered_data[filtered_data['experience'] <= max_experience]

#     # Apply Boolean search for skills
#     if mandatory_skills:
#         for skill in mandatory_skills:
#             filtered_data = filtered_data[filtered_data['skills'].str.contains(skill, case=False)]

#     if optional_skills:
#         optional_filter = filtered_data['skills'].str.contains('|'.join(optional_skills), case=False)
#         filtered_data = filtered_data[optional_filter]

#     if class_model is None or reg_model is None:
#         print("Error: Models were not successfully trained. Cannot perform predictions.")
#         return None

#     # Add 'Job_Role_Skill_Category' if it doesn't exist
#     if 'Job_Role_Skill_Category' not in filtered_data.columns:
#         filtered_data['Job_Role_Skill_Category'] = required_role

#     X_new = filtered_data[['skills', 'experience', 'Job_Role_Skill_Category']]
#     classification_predictions = class_model.predict(X_new)
#     salary_predictions = reg_model.predict(X_new)

#     filtered_data['Classification_Prediction'] = classification_predictions
#     filtered_data['Predicted_Salary'] = salary_predictions

#     # Sort by Classification_Prediction and Predicted_Salary
#     top_candidates = filtered_data.sort_values(['Classification_Prediction', 'Predicted_Salary'], ascending=[False, False]).head(top_n)

#     desired_fields = [
#         'candidate_id', 'full_name', 'email', 'candidate_status', 'experience',
#         'profile_title', 'source_name', 'date', 'Classification_Prediction',
#         'Predicted_Salary', 'skills'
#     ]

#     available_fields = [field for field in desired_fields if field in top_candidates.columns]
    
#     # Extract top 5 skills for each candidate
#     def extract_top_skills(skills_str, n=5):
#         skills = skills_str.split(',')
#         return ','.join(skills[:n])

#     top_candidates['skills'] = top_candidates['skills'].apply(extract_top_skills)

#     return top_candidates[available_fields]

# @app.route('/rank-candidates', methods=['POST'])
# def rank_candidates():
#     try:
#         data = request.json
#         required_role = data.get('requiredRole')
#         top_n = int(data.get('topN', 5))
#         main_data_file_path = data.get('mainDataFilePath')
#         job_roles_data_file_path = data.get('jobRolesDataFilePath')
#         test_data_file_path = data.get('testDataFilePath')
#         start_date_str = data.get('startDate')
#         end_date_str = data.get('endDate')
#         max_experience = data.get('maxExperience')
#         mandatory_skills = data.get('mandatorySkills', [])
#         optional_skills = data.get('optionalSkills', [])

#         print(f"Processing request for role: {required_role}, top {top_n} candidates")
#         print(f"Date filter: Start Date = {start_date_str}, End Date = {end_date_str}")
#         print(f"Max Experience: {max_experience}")
#         print(f"Mandatory Skills: {mandatory_skills}")
#         print(f"Optional Skills: {optional_skills}")

#         main_df = load_and_prepare_data(main_data_file_path)
#         job_roles_df = load_and_prepare_data(job_roles_data_file_path, is_job_roles=True)
#         if main_df is None or job_roles_df is None:
#             return jsonify({"error": "Failed to load main or job roles data."}), 500

#         main_df = engineer_features(main_df)
#         job_roles_df = engineer_features(job_roles_df)
#         extracted_skills = extract_skills_from_training_data(main_df, job_roles_df)
#         trained_class_model, trained_reg_model = train_models(main_df, job_roles_df)
#         if trained_class_model is None or trained_reg_model is None:
#             return jsonify({"error": "Failed to train models"}), 500

#         new_df = load_and_prepare_data(test_data_file_path, parse_dates=True)
#         if new_df is None:
#             return jsonify({"error": "Failed to load test data."}), 500

#         if 'date' not in new_df.columns:
#             return jsonify({"error": "Test data must contain a 'date' column."}), 400

#         apply_date_filter = False
#         if start_date_str and end_date_str:
#             apply_date_filter = True
#             try:
#                 start_date = pd.to_datetime(start_date_str)
#                 end_date = pd.to_datetime(end_date_str)
#             except Exception as e:
#                 return jsonify({"error": f"Invalid date format: {e}"}), 400
#             if start_date > end_date:
#                 return jsonify({"error": "'startDate' cannot be after 'endDate'."}), 400
#         elif start_date_str or end_date_str:
#             return jsonify({"error": "Both 'startDate' and 'endDate' must be provided if filtering by date."}), 400

#         if apply_date_filter:
#             new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
#             before_filter_count = len(new_df)
#             new_df = new_df[(new_df['date'] >= start_date) & (new_df['date'] <= end_date)]
#             after_filter_count = len(new_df)
#             print(f"Filtered test data from {before_filter_count} to {after_filter_count} records based on date.")
#             if new_df.empty:
#                 return jsonify({"error": "No candidates found within the specified date range."}), 404
#         else:
#             print("No date filter applied.")

#         new_df = engineer_features(new_df)
#         results = predict_and_rank(
#             trained_class_model, trained_reg_model, new_df, required_role,
#             extracted_skills, top_n, max_experience, mandatory_skills, optional_skills
#         )

#         if results is not None and not results.empty:
#             if 'date' in results.columns:
#                 results['date'] = results['date'].dt.strftime('%Y-%m-%d')
#             results_dict = results.to_dict(orient='records')
#             return jsonify(results_dict)
#         else:
#             return jsonify({"error": "No results found or an error occurred"}), 404

#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         traceback.print_exc()
#         return jsonify({"error": "An unexpected error occurred."}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)