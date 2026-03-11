import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Attempting to load the dataset...")
dataset_file = 'IMDb Movies India.csv'

if not os.path.exists(dataset_file):
    zip_file = 'IMDb Movies India.csv.zip'
    if os.path.exists(zip_file):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()
        print("Dataset extracted successfully.")
    else:
        print(f"\n{zip_file} not found. Creating sample dataset for demonstration...")
        np.random.seed(42)
        sample_data = {
            'Name': [f'Movie {i}' for i in range(1, 101)],
            'Year': np.random.randint(2000, 2024, 100),
            'Duration': np.random.randint(90, 180, 100),
            'Genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Thriller', 'Romance'], 100),
            'Rating': np.random.uniform(5, 9, 100),
            'Votes': np.random.randint(1000, 2000000, 100),
            'Director': np.random.choice(['Director A', 'Director B', 'Director C', 'Director D', 'Director E'], 100),
            'Actor 1': np.random.choice(['Actor A', 'Actor B', 'Actor C', 'Actor D'], 100),
            'Actor 2': np.random.choice(['Actor E', 'Actor F', 'Actor G', 'Actor H'], 100),
            'Actor 3': np.random.choice(['Actor I', 'Actor J', 'Actor K', 'Actor L'], 100),
        }
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(dataset_file, index=False)
        print(f"Sample dataset created: {dataset_file}")

print("\nLoading the dataset...")
df = pd.read_csv(dataset_file)
print("Dataset loaded successfully.")

print("\nData Exploration:")
print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())
print("Basic statistics:\n", df.describe())

print("\nGenerating missing values heatmap...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('missing_values_heatmap.png', dpi=100, bbox_inches='tight')
print("Heatmap saved as 'missing_values_heatmap.png'")

print("\nData Cleaning:")
important_columns = ['Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes']
df.dropna(subset=important_columns, inplace=True)
print("Dropped rows with missing values in important columns.")
print("New dataset shape:", df.shape)

print("\nFeature Engineering:")
le_genre = LabelEncoder()
df['Genre_encoded'] = le_genre.fit_transform(df['Genre'])

le_director = LabelEncoder()
df['Director_encoded'] = le_director.fit_transform(df['Director'])

le_actor1 = LabelEncoder()
df['Actor1_encoded'] = le_actor1.fit_transform(df['Actor 1'])

le_actor2 = LabelEncoder()
df['Actor2_encoded'] = le_actor2.fit_transform(df['Actor 2'])

le_actor3 = LabelEncoder()
df['Actor3_encoded'] = le_actor3.fit_transform(df['Actor 3'])

# Scaling numerical features
scaler = StandardScaler()
df[['Duration_scaled', 'Votes_scaled']] = scaler.fit_transform(df[['Duration', 'Votes']])

print("Categorical features encoded and numerical features scaled.")

# Feature Selection
features = ['Genre_encoded', 'Director_encoded', 'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded', 'Duration_scaled', 'Votes_scaled']
X = df[features]
y = df['Rating']

# Split the dataset
print("\nSplitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")

# Model Training
print("\nModel Training:")
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression model trained.")

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Regressor model trained.")

# Model Evaluation
print("\nModel Evaluation:")
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    return mae, mse, r2

lr_mae, lr_mse, lr_r2 = evaluate_model(lr, X_test, y_test, "Linear Regression")
rf_mae, rf_mse, rf_r2 = evaluate_model(rf, X_test, y_test, "Random Forest Regressor")

# Compare models
print("\nModel Comparison:")
if lr_r2 > rf_r2:
    best_model = lr
    best_model_name = "Linear Regression"
else:
    best_model = rf
    best_model_name = "Random Forest Regressor"
print(f"Best model based on R2 score: {best_model_name}")

# Model Saving
print("\nSaving the best model and encoders...")
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(le_genre, 'le_genre.pkl')
joblib.dump(le_director, 'le_director.pkl')
joblib.dump(le_actor1, 'le_actor1.pkl')
joblib.dump(le_actor2, 'le_actor2.pkl')
joblib.dump(le_actor3, 'le_actor3.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models and encoders saved successfully.")

# Prediction Function
def predict_rating(genre, director, actor1, actor2, actor3, duration, votes):
    """
    Predict the movie rating based on input features.

    Parameters:
    genre (str): Movie genre
    director (str): Movie director
    actor1 (str): Lead actor 1
    actor2 (str): Lead actor 2
    actor3 (str): Lead actor 3
    duration (int): Movie duration in minutes
    votes (int): Number of votes

    Returns:
    float: Predicted rating (or error message if label not found)
    """
    try:
        genre_enc = le_genre.transform([genre])[0]
        director_enc = le_director.transform([director])[0]
        actor1_enc = le_actor1.transform([actor1])[0]
        actor2_enc = le_actor2.transform([actor2])[0]
        actor3_enc = le_actor3.transform([actor3])[0]
        scaled_features = scaler.transform([[duration, votes]])
        duration_scaled = scaled_features[0][0]
        votes_scaled = scaled_features[0][1]
        features = [[genre_enc, director_enc, actor1_enc, actor2_enc, actor3_enc, duration_scaled, votes_scaled]]
        return best_model.predict(features)[0]
    except ValueError as e:
        return f"Error: One or more input values not found in training data. {str(e)}"

# Example Prediction - Using values from the training dataset
print("\nExample Prediction:")
# Get actual values from the dataset for demonstration
sample_genre = df['Genre'].iloc[0]
sample_director = df['Director'].iloc[0]
sample_actor1 = df['Actor 1'].iloc[0]
sample_actor2 = df['Actor 2'].iloc[0]
sample_actor3 = df['Actor 3'].iloc[0]
sample_duration = df['Duration'].iloc[0]
sample_votes = df['Votes'].iloc[0]

predicted_rating = predict_rating(sample_genre, sample_director, sample_actor1, sample_actor2, sample_actor3, sample_duration, sample_votes)
print(f"Predicted rating for sample movie: {predicted_rating:.2f}")
print(f"(Genre: {sample_genre}, Director: {sample_director}, Duration: {sample_duration} min, Votes: {sample_votes})")

print("\nProject completed successfully!")