# for data manipulation
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/vijeee/PIMA-Diabetes-Prediction/Xtrain.csv"                    # enter the Hugging Face username here
Xtest_path = "hf://datasets/vijeee/PIMA-Diabetes-Prediction/Xtest.csv"                      # enter the Hugging Face username here
ytrain_path = "hf://datasets/vijeee/PIMA-Diabetes-Prediction/ytrain.csv"                    # enter the Hugging Face username here
ytest_path = "hf://datasets/vijeee/PIMA-Diabetes-Prediction/ytest.csv"                      # enter the Hugging Face username here

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# scale numeric features
numeric_features = [
    'preg',
    'plas',
    'pres',
    'skin',
    'test',
    'mass',
    'pedi',
    'age'
]


# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define GB model
gb_model = GradientBoostingClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'gradientboostingclassifier__n_estimators': [75, 100, 125],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__subsample': [0.5, 0.6]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, gb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)


# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_pima_diabetes_model_v1.joblib")

# Upload to Hugging Face
repo_id = "vijeee/PIMA-Diabetes-Prediction"                                         # enter the Hugging Face username here
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_pima_diabetes_model_v1.joblib",
    path_in_repo="best_pima_diabetes_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
