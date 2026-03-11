# Titanic Survival Prediction - Machine Learning Model

A comprehensive machine learning project that predicts whether a passenger would have survived the Titanic disaster using a Random Forest classifier.

## Project Overview

This project demonstrates the complete machine learning pipeline:
- **Data Loading**: Download Titanic dataset from Kaggle using `kagglehub`
- **Data Preprocessing**: Handle missing values and convert categorical variables
- **Model Training**: Train a Random Forest classifier
- **Model Evaluation**: Calculate accuracy, confusion matrix, and classification report
- **Prediction**: Make predictions for new passenger data

## Features

✅ **End-to-end ML Pipeline**: Complete workflow from data loading to predictions
✅ **Data Preprocessing**: Proper handling of missing values and categorical encoding
✅ **Random Forest Classifier**: Robust ensemble learning algorithm
✅ **Comprehensive Evaluation**: Accuracy, confusion matrix, and classification metrics
✅ **Prediction Function**: Predict survival for new passengers
✅ **Feature Engineering**: Uses 7 relevant features for prediction
✅ **Clean Code Structure**: Well-organized sections with clear documentation

## Project Structure

```
titan/
├── titanic_survival_prediction.py  # Main Python script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.7 or higher
- Internet connection (for downloading the Kaggle dataset)
- Kaggle account (for using kagglehub)

## Installation

### 1. Set Up Python Environment

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle Credentials

The `kagglehub` library requires Kaggle credentials:

1. Visit [Kaggle Settings](https://www.kaggle.com/settings/account)
2. Scroll to "API" section and click "Create New Token"
3. This downloads a `kaggle.json` file
4. Place `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<YourUsername>\.kaggle\`)
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (macOS/Linux only)

## Usage

### Run the Complete Pipeline

```bash
python titanic_survival_prediction.py
```

The script will:
1. Load the Titanic dataset from Kaggle
2. Preprocess the data (handle missing values, encode categorical variables)
3. Split data into training and testing sets (80-20 split)
4. Train a Random Forest classifier
5. Evaluate the model with metrics and confusion matrix
6. Display predictions for sample passengers
7. Print final accuracy and summary

### Example Output

```
TITANIC SURVIVAL PREDICTION - MACHINE LEARNING MODEL

STEP 1: DATASET LOADING
Dataset shape: (891, 12)
...

STEP 2: DATA PREPROCESSING
Handling missing values...
✓ Missing values handled!
...

STEP 3: DATA SPLITTING
Training set size: (712, 7)
Testing set size: (179, 7)
...

STEP 4: MODEL TRAINING
✓ Model trained successfully!
...

STEP 5: MODEL EVALUATION
Training Accuracy: 0.9831 (98.31%)
Testing Accuracy: 0.8268 (82.68%)
...

STEP 6: PREDICTION FOR NEW PASSENGERS
Passenger 1 - Female, First Class:
  Prediction: Survived
  Survival Probability: 95.43%
...

SUMMARY
✓ Model testing accuracy: 0.8268 (82.68%)
```

## Model Details

### Features Used

1. **Pclass**: Passenger class (1, 2, or 3)
2. **Sex**: Gender (male/female) - encoded as 1/0
3. **Age**: Age in years
4. **SibSp**: Number of siblings/spouses aboard
5. **Parch**: Number of parents/children aboard
6. **Fare**: Ticket fare in pounds
7. **Embarked**: Port of embarkation (S/C/Q) - encoded as 0/1/2

### Target Variable

- **Survived**: 0 = Did not survive, 1 = Survived

### Model Configuration

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Train-Test Split**: 80-20

## Making Predictions

The script includes example predictions for different types of passengers:

```python
passenger = {
    'Pclass': 1,        # Passenger class
    'Sex': 'female',    # Gender (male or female)
    'Age': 25,          # Age in years
    'SibSp': 1,         # Siblings/spouses aboard
    'Parch': 0,         # Parents/children aboard
    'Fare': 100,        # Ticket fare
    'Embarked': 'S'     # Embarkation port (S/C/Q)
}

result = predict_survival(model, passenger)
print(f"Prediction: {result['prediction_text']}")
print(f"Probability: {result['survival_probability']:.2%}")
```

## Evaluation Metrics

### Accuracy
Percentage of correct predictions out of all predictions made.

### Confusion Matrix
Shows the distribution of true positives, true negatives, false positives, and false negatives.

### Classification Report
Includes:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

## Key Insights

1. **Gender Bias**: Females had a significantly higher survival rate
2. **Class Matters**: First-class passengers had better survival chances
3. **Age Factor**: Children and younger passengers had higher survival rates
4. **Family Size**: Traveling with family members affected survival chances
5. **Fare Impact**: Passengers who paid more had better survival chances

## Troubleshooting

### Issue: Kaggle credentials not found
**Solution**: Ensure `kaggle.json` is in the correct location (`~/.kaggle/`)

### Issue: PyArrow import error
**Solution**: 
```bash
pip install pyarrow
```

### Issue: Dataset not downloading
**Solution**: 
- Check internet connection
- Verify Kaggle credentials are correct
- Ensure kagglehub is installed: `pip install kagglehub`

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **kagglehub**: Download datasets from Kaggle

## Author

Created as a comprehensive example of machine learning pipeline implementation.

## License

Free to use and modify for educational purposes.

## Further Improvements

Potential enhancements:
- Try other algorithms (Logistic Regression, Decision Tree, XGBoost)
- Implement hyperparameter tuning with GridSearchCV
- Add feature scaling/normalization
- Perform cross-validation
- Create a web interface with Flask/Streamlit
- Save the trained model for reuse
- Implement ensemble methods
- Add more feature engineering techniques

---

**Happy Learning!** 🚢📊🤖
