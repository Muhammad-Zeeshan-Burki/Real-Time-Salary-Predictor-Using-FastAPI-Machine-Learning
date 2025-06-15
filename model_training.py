import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pickle
import os

print("Starting data preprocessing and model training...")

# Define the path to the dataset
try:
    df = pd.read_csv('ai_job_dataset.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    
    # Map the dataset columns to the expected features for our model
    # Based on the dataset structure, we'll select relevant columns
    df_processed = pd.DataFrame({
        'Job_Title': df['job_title'],
        'Experience': df['years_experience'], 
        'Education_Level': df['education_required'],
        'Employment_Type': df['employment_type'].map({
            'FL': 'Full-time',
            'PT': 'Part-time', 
            'CT': 'Contract',
            'FT': 'Full-time'
        }).fillna('Full-time'),
        'City': df['company_location'],
        'Programming_Language': df['required_skills'].str.split(',').str[0].str.strip(),  # Take first skill as primary language
        'Salary_in_USD': df['salary_usd']
    })
    
    # Clean and filter data
    df_processed = df_processed.dropna()
    df_processed = df_processed[df_processed['Salary_in_USD'] > 0]
    
    df = df_processed
    print(f"Processed dataset shape: {df.shape}")
    
except FileNotFoundError:
    print("Error: ai_job_dataset.csv not found. Please ensure the file is in the correct directory.")
    # Create a dummy DataFrame for demonstration if file not found
    data = {
        'Job_Title': ['Data Scientist', 'AI Engineer', 'Machine Learning Engineer', 'Data Analyst', 'Research Scientist'],
        'Experience': [5, 3, 7, 2, 6],
        'Education_Level': ['Masters', 'Bachelors', 'PhD', 'Bachelors', 'Masters'],
        'Employment_Type': ['Full-time', 'Full-time', 'Full-time', 'Part-time', 'Full-time'],
        'City': ['New York', 'San Francisco', 'London', 'Berlin', 'Paris'],
        'Programming_Language': ['Python', 'Python', 'Python', 'R', 'Python'],
        'Salary_in_USD': [120000, 110000, 150000, 70000, 130000]
    }
    df = pd.DataFrame(data)
    print("Using dummy data for demonstration.")

# Drop unnecessary columns if any (e.g., 'Company_Name' if not used as a feature)
# For this dataset, all columns seem relevant or can be encoded.
# Let's check for 'Company_Name' and similar less relevant columns based on common salary prediction datasets.
# If 'Company_Name' exists and has too many unique values, it might be dropped.
if 'Company_Name' in df.columns:
    df = df.drop('Company_Name', axis=1)
    print("Dropped 'Company_Name' column.")

# Rename columns for easier access, if necessary (e.g., if there are spaces/special chars)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')

# Target variable and features
TARGET = 'Salary_in_USD'
FEATURES = [col for col in df.columns if col != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"Features: {FEATURES}")
print(f"Target: {TARGET}")

# Identify categorical and numerical features
# Automatically detect based on data type, or provide a manual list if more control is needed.
categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # 'ignore' handles unseen categories in prediction

# Create a column transformer to apply different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) as they are
)

# Create the full pipeline with preprocessing and the model
# RandomForestRegressor is chosen for its good performance and robustness
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 uses all available cores
])

print("Splitting data into training and testing sets...")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")


print("Training the model...")
# Train the model
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

print("Evaluating the model...")
# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (R2): {r2:.4f}")

# Save the trained model and the feature names for later use in FastAPI
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Save as both joblib and pickle format
model_path_joblib = os.path.join(model_dir, 'salary_predictor_model.joblib')
model_path_pkl = os.path.join(model_dir, 'salary_predictor_model.pkl')

joblib.dump(model_pipeline, model_path_joblib)
print(f"Model saved to {model_path_joblib}")

with open(model_path_pkl, 'wb') as f:
    pickle.dump(model_pipeline, f)
print(f"Model saved to {model_path_pkl}")

# Save the list of feature names used for training.
# This ensures that the order of features is consistent during prediction.
original_feature_names_path_joblib = os.path.join(model_dir, 'original_feature_names.joblib')
original_feature_names_path_pkl = os.path.join(model_dir, 'original_feature_names.pkl')

joblib.dump(FEATURES, original_feature_names_path_joblib)
print(f"Original feature names saved to {original_feature_names_path_joblib}")

with open(original_feature_names_path_pkl, 'wb') as f:
    pickle.dump(FEATURES, f)
print(f"Original feature names saved to {original_feature_names_path_pkl}")

print("Preprocessing and model training script finished.")
