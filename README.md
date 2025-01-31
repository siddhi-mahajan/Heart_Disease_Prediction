# HEART DISEASE RISK PREDICTION

This project aims to predict the risk of heart disease using machine learning algorithms. By analyzing key health indicators, such as cholesterol levels, blood pressure, age, and other features, the system provides an estimate of whether a patient is at risk of developing heart disease.

## TABLE OF CONTENT
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Contribution](#contribution)

## FEATURES
- **Data Preprocessing:** Cleansing and transforming the dataset to handle missing values and standardize inputs.
- **Machine Learning:** Implemented various classification algorithms (e.g., Logistic Regression, Random Forest, K-Nearest Neighbors) to train and evaluate the model on a heart disease dataset.
- **Model Evaluation:** Used metrics like accuracy, precision, recall, and F1-score to evaluate the model's performance.
- **Flask API:** A simple web interface is created using Flask to allow users to input patient data and get predictions on heart disease risk.
- **Google Colab Integration:** The machine learning model is trained and developed in Google Colab for ease of use and rapid prototyping.

## TECHNOLOGIES USED
- **Python:** Core programming language for data analysis and machine learning.
- **Pandas & NumPy:**  For data manipulation and preprocessing.
- **Scikit-learn:** For building and evaluating machine learning models.
- **Google Colab:** Used for developing and training the model in a cloud-based environment.
- **Flask:** For building the web API that serves the prediction model.

## HOW IT WORKS
- **Train the Model:** The dataset is loaded into Google Colab where it undergoes preprocessing and model training.
- **Deploy with Flask:** The trained model is deployed via a Flask web server. Users can input patient details via a form, and the API returns the prediction.
- **User Interface:** A simple HTML form collects patient data such as age, cholesterol, and blood pressure, which is sent to the Flask API for prediction.

## INSTALLATION
### STEP1: Clone the Repository
```bash
git clone https://github.com/siddhi-mahajan/heart-disease-risk-prediction.git
```
### STEP2: Install Dependencies
```bash 
pip install -r requirements.txt
```
### STEP3: Run the Flask Application
```bash
flask run
```

## CONTRIBUTIONS
Contributions, issues, and feature requests are welcome. Feel free to fork the repository and submit pull requests.


