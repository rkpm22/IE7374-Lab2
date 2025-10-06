# IE7374-Lab2: Student Performance Linear Regression ML Pipeline with Airflow

This project demonstrates how to orchestrate a **Linear Regression** machine learning pipeline using Apache Airflow to predict student performance. The pipeline processes student data, handles categorical variables, trains a linear regression model, and makes predictions.

## Project Structure

- **`dags/airflow.py`** - Airflow DAG orchestrating the ML pipeline
- **`dags/src/lab.py`** - Linear regression implementation functions
- **`dags/data/`** - Training and test datasets
- **`docker-compose.yaml`** - Airflow cluster setup
- **`setup.sh`** - Environment initialization script

## ML Pipeline Overview

1. **Data Loading**: Loads Student Performance dataset (10,001 records)
2. **Data Preprocessing**: Cleans data, handles categorical variables (Yes/No → 1/0), applies scaling
3. **Model Training**: Trains Linear Regression model with train/test split (80/20)
4. **Model Evaluation**: Calculates R² and MSE metrics for performance assessment
5. **Prediction**: Makes predictions on sample student test cases

## Student Performance Dataset

### Dataset Features
The pipeline uses `Student_Performance.csv` with:
- **Hours Studied**: Number of hours studied (continuous)
- **Previous Scores**: Previous academic scores (continuous)
- **Extracurricular Activities**: Yes/No (categorical → encoded as 1/0)
- **Sleep Hours**: Hours of sleep (continuous)
- **Sample Question Papers Practiced**: Number of practice papers (continuous)
- **Performance Index**: Target variable (0-100 scale)

### Dataset Statistics
- **Total Records**: 10,001 students
- **Target Range**: Performance Index 0-100
- **Categorical Handling**: Yes/No automatically converted to 1/0
- **Preprocessing**: Standard scaling applied to all features

## Usage

1. **Dataset**: The `Student_Performance.csv` is already included
2. **Initialize environment**: `./setup.sh`
3. **Start Airflow**: `docker compose up`
4. **Access UI**: Open `http://localhost:8080` (username: airflow2, password: airflow2)
5. **Run DAG**: Trigger `Linear_Regression_DAG` manually
6. **Monitor**: Check logs and task execution in the UI

## Model Outputs

- **Model file**: `model/model.sav` - Trained linear regression model
- **Scaler**: `model/scaler.sav` - Feature scaling parameters
- **Label Encoder**: `model/label_encoder.sav` - Categorical encoding (Yes/No → 1/0)
- **Metrics**: Training and test R², MSE scores
- **Predictions**: Performance Index predictions for 5 sample students

## Sample Predictions

The pipeline creates test cases for 5 students with different profiles:
- **Student 1**: 6 hours studied, 85 previous score, Yes extracurricular, 8 sleep hours, 3 practice papers
- **Student 2**: 8 hours studied, 92 previous score, No extracurricular, 7 sleep hours, 5 practice papers
- And more...

## Key Features

- **Real-world dataset**: 10,001 student records with realistic features
- **Categorical handling**: Automatic Yes/No → 1/0 encoding
- **Standard scaling** for optimal linear regression performance
- **Train/test split** (80/20) for proper evaluation
- **Comprehensive metrics** (R², MSE) for model assessment
- **Sample predictions**: 5 diverse student test cases
- **Scalable architecture** with Docker
- **XCom data passing** between tasks