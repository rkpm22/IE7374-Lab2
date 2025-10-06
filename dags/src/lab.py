import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import base64
import numpy as np

def load_data():
    """
    Loads Student Performance dataset for linear regression.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading Student Performance dataset for linear regression")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/Student_Performance.csv"))
    print(f"Dataset loaded with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target variable: Performance Index")
    print(f"Features: {list(df.columns[:-1])}")
    
    # Show basic statistics
    print(f"Target (Performance Index) - Min: {df['Performance Index'].min():.1f}, Max: {df['Performance Index'].max():.1f}, Mean: {df['Performance Index'].mean():.1f}")
    
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing for linear regression,
    including handling categorical variables.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    print(f"Original dataset shape: {df.shape}")
    
    # Clean data
    df = df.dropna()
    print(f"After dropping NaN: {df.shape}")
    
    # For linear regression, we need features (X) and target (y)
    # Performance Index is the target variable
    target_column = 'Performance Index'
    feature_columns = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
    
    print(f"Target column: {target_column}")
    print(f"Feature columns: {feature_columns}")
    
    # Prepare features
    X = df[feature_columns].copy()
    y = df[target_column]
    
    # Handle categorical variable (Extracurricular Activities: Yes/No -> 1/0)
    label_encoder = LabelEncoder()
    X['Extracurricular Activities'] = label_encoder.fit_transform(X['Extracurricular Activities'])
    print(f"Categorical encoding - Yes: 1, No: 0")
    
    # Scale features for better linear regression performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Feature statistics after preprocessing:")
    for i, col in enumerate(feature_columns):
        print(f"  {col}: Mean={X_scaled[:, i].mean():.3f}, Std={X_scaled[:, i].std():.3f}")
    
    # Create a dictionary to store both X and y
    processed_data = {
        'X': X_scaled,
        'y': y.values,
        'feature_names': feature_columns,
        'target_name': target_column,
        'scaler': scaler,
        'label_encoder': label_encoder
    }

    # bytes -> base64 string for XCom
    processed_serialized_data = pickle.dumps(processed_data)
    return base64.b64encode(processed_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a Linear Regression model on the preprocessed data and saves it.
    Returns the model metrics (JSON-serializable).
    """
    # decode -> bytes -> processed data dictionary
    data_bytes = base64.b64decode(data_b64)
    processed_data = pickle.loads(data_bytes)

    X = processed_data['X']
    y = processed_data['y']
    
    print(f"Training data shape - X: {X.shape}, y: {y.shape}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_names': processed_data['feature_names'],
        'target_name': processed_data['target_name']
    }
    
    print(f"Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

    # Save model, scaler, and label encoder
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.sav")
    with open(scaler_path, "wb") as f:
        pickle.dump(processed_data['scaler'], f)
    
    # Save label encoder
    encoder_path = os.path.join(output_dir, "label_encoder.sav")
    with open(encoder_path, "wb") as f:
        pickle.dump(processed_data['label_encoder'], f)

    return metrics  # dict is JSON-safe


def load_model_predict(filename: str, metrics: dict):
    """
    Loads the saved model and makes predictions on test data.
    Creates sample test data for student performance prediction.
    """
    # Load the saved model
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(model_path, "rb"))
    
    # Load the scaler
    scaler_path = os.path.join(os.path.dirname(__file__), "../model", "scaler.sav")
    scaler = pickle.load(open(scaler_path, "rb"))
    
    # Load the label encoder
    encoder_path = os.path.join(os.path.dirname(__file__), "../model", "label_encoder.sav")
    label_encoder = pickle.load(open(encoder_path, "rb"))

    # Create sample test data for student performance prediction
    test_data = {
        'Hours Studied': [6, 8, 4, 9, 3],
        'Previous Scores': [85, 92, 78, 95, 65],
        'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'Sleep Hours': [8, 7, 6, 9, 5],
        'Sample Question Papers Practiced': [3, 5, 2, 6, 1]
    }
    
    test_df = pd.DataFrame(test_data)
    print(f"Created test data with shape: {test_df.shape}")
    print(f"Test data:")
    print(test_df)
    
    # Prepare test data with same preprocessing as training
    feature_names = metrics['feature_names']
    X_test = test_df[feature_names].copy()
    
    # Encode categorical variable
    X_test['Extracurricular Activities'] = label_encoder.transform(X_test['Extracurricular Activities'])
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    predictions = loaded_model.predict(X_test_scaled)
    
    print(f"\nModel performance summary:")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Training MSE: {metrics['train_mse']:.4f}")
    print(f"Test MSE: {metrics['test_mse']:.4f}")
    
    print(f"\nPredictions on test students:")
    for i, pred in enumerate(predictions):
        print(f"  Student {i+1}: Predicted Performance Index = {pred:.1f}")
        print(f"    Hours Studied: {test_df.iloc[i]['Hours Studied']}, "
              f"Previous Scores: {test_df.iloc[i]['Previous Scores']}, "
              f"Extracurricular: {test_df.iloc[i]['Extracurricular Activities']}, "
              f"Sleep Hours: {test_df.iloc[i]['Sleep Hours']}, "
              f"Practice Papers: {test_df.iloc[i]['Sample Question Papers Practiced']}")

    # Return all predictions and performance metrics
    result = {
        'predictions': [float(pred) for pred in predictions],
        'test_data': test_df.to_dict('records'),
        'model_performance': metrics
    }
    
    return result
