"""
Titanic Survival Prediction using Machine Learning
This script builds a classification model to predict whether a passenger survived the Titanic disaster.
"""

import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_dataset():
    """
    Download and load the Titanic dataset from Kaggle using kagglehub.
    
    Returns:
        pd.DataFrame: The loaded Titanic dataset
    """
    print("=" * 70)
    print("STEP 1: DATASET LOADING")
    print("=" * 70)
    
    try:
        file_path = "Titanic-Dataset.csv"
        
        print(f"Loading Titanic dataset from Kaggle...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "yasserh/titanic-dataset",
            file_path,
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst 5 rows of the dataset:")
        print(df.head())
        print(f"\nDataset info:")
        print(df.info())
        print("\n")
        
        return df
    
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Make sure you have kagglehub installed and your Kaggle credentials configured.")
        raise


def preprocess_data(df):
    """
    Preprocess the Titanic dataset:
    - Handle missing values
    - Convert categorical variables to numerical
    - Select relevant features
    
    Returns:
        tuple: (X - features, y - target, feature_names - list of feature names)
    """
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)

    data = df.copy()

    print("Missing values before preprocessing:")
    print(data.isnull().sum())
    print()

    print("Handling missing values...")
    data['Age'].fillna(data['Age'].median(), inplace=True)

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    data.dropna(subset=['Fare'], inplace=True)

    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    print("✓ Missing values handled!")
    print()

    print("Selecting relevant features...")
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[features].copy()
    y = data['Survived'].copy()
    
    print(f"✓ Features selected: {features}")
    print()

    print("Converting categorical variables to numerical...")

    le_sex = LabelEncoder()
    X['Sex'] = le_sex.fit_transform(X['Sex'])
    print(f"  - Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

    le_embarked = LabelEncoder()
    X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
    print(f"  - Embarked encoding: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")
    
    print("✓ Categorical variables converted!")
    print()

    print("Preprocessed feature statistics:")
    print(X.describe())
    print()

    print("Final check for missing values:")
    print(X.isnull().sum())
    print()
    
    return X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("=" * 70)
    print("STEP 3: DATA SPLITTING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print()
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest classification model.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        RandomForestClassifier: Trained model
    """
    print("=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully!")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print()
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model using accuracy, confusion matrix, and classification report.
    
    Args:
        model: Trained classification model
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
    """
    print("=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print()

    print("Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    print()

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['Did Not Survive', 'Survived']))
    print()

    print("Feature Importance:")
    feature_importance = model.feature_importances_
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for feature, importance in sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True):
        print(f"  - {feature}: {importance:.4f}")
    print()
    
    return test_accuracy

def predict_survival(model, passenger_data):
    """
    Predict whether a new passenger would have survived.
    
    Args:
        model: Trained classification model
        passenger_data: Dictionary with passenger information
                       Keys: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
    
    Returns:
        dict: Prediction result with survival status and probability
    """

    required_keys = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if not all(key in passenger_data for key in required_keys):
        return {"error": f"Missing required keys. Required: {required_keys}"}

    data_dict = passenger_data.copy()

    sex_mapping = {'male': 1, 'female': 0, 'Male': 1, 'Female': 0}
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    
    if isinstance(data_dict['Sex'], str):
        data_dict['Sex'] = sex_mapping.get(data_dict['Sex'], 1)
    
    if isinstance(data_dict['Embarked'], str):
        data_dict['Embarked'] = embarked_mapping.get(data_dict['Embarked'], 0)

    features_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X_new = np.array([data_dict[feature] for feature in features_order]).reshape(1, -1)

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]
    
    result = {
        'survived': bool(prediction),
        'survival_probability': probability[1],
        'did_not_survive_probability': probability[0],
        'prediction_text': 'Survived' if prediction == 1 else 'Did Not Survive'
    }
    
    return result

def main():
    """Main execution flow."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  TITANIC SURVIVAL PREDICTION - MACHINE LEARNING MODEL".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    df = load_dataset()

    X, y, features = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    test_accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)

    print("=" * 70)
    print("STEP 6: PREDICTION FOR NEW PASSENGERS")
    print("=" * 70)
    print()

    passenger_1 = {
        'Pclass': 1,      
        'Sex': 'female',  
        'Age': 25,        
        'SibSp': 1,       
        'Parch': 0,       
        'Fare': 100,      
        'Embarked': 'S'   
    }
    
    print("Passenger 1 - Female, First Class:")
    print(f"  Pclass: {passenger_1['Pclass']}, Sex: {passenger_1['Sex']}, Age: {passenger_1['Age']}")
    print(f"  SibSp: {passenger_1['SibSp']}, Parch: {passenger_1['Parch']}, Fare: {passenger_1['Fare']}, Embarked: {passenger_1['Embarked']}")
    pred_1 = predict_survival(model, passenger_1)
    print(f"  Prediction: {pred_1['prediction_text']}")
    print(f"  Survival Probability: {pred_1['survival_probability']:.2%}")
    print()
    
    passenger_2 = {
        'Pclass': 3,      
        'Sex': 'male',    
        'Age': 35,        
        'SibSp': 0,       
        'Parch': 0,       
        'Fare': 7.75,     
        'Embarked': 'S'   
    }
    
    print("Passenger 2 - Male, Third Class:")
    print(f"  Pclass: {passenger_2['Pclass']}, Sex: {passenger_2['Sex']}, Age: {passenger_2['Age']}")
    print(f"  SibSp: {passenger_2['SibSp']}, Parch: {passenger_2['Parch']}, Fare: {passenger_2['Fare']}, Embarked: {passenger_2['Embarked']}")
    pred_2 = predict_survival(model, passenger_2)
    print(f"  Prediction: {pred_2['prediction_text']}")
    print(f"  Survival Probability: {pred_2['survival_probability']:.2%}")
    print()
    
    passenger_3 = {
        'Pclass': 1,      
        'Sex': 'male',    
        'Age': 5,         
        'SibSp': 1,       
        'Parch': 2,       
        'Fare': 151.83,   
        'Embarked': 'S'   
    }
    
    print("Passenger 3 - Child, First Class:")
    print(f"  Pclass: {passenger_3['Pclass']}, Sex: {passenger_3['Sex']}, Age: {passenger_3['Age']}")
    print(f"  SibSp: {passenger_3['SibSp']}, Parch: {passenger_3['Parch']}, Fare: {passenger_3['Fare']}, Embarked: {passenger_3['Embarked']}")
    pred_3 = predict_survival(model, passenger_3)
    print(f"  Prediction: {pred_3['prediction_text']}")
    print(f"  Survival Probability: {pred_3['survival_probability']:.2%}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Model testing accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"✓ Algorithm used: Random Forest Classifier")
    print(f"✓ Number of features: {len(features)}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    print()
    print("Model training and evaluation completed successfully!")
    print("\n")


if __name__ == "__main__":
    main()
