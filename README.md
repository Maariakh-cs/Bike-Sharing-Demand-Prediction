# 🚲 Bike Sharing Demand Prediction using Machine Learning

An end-to-end Machine Learning project that predicts **hourly bike rental demand** based on weather conditions, seasonal patterns, and time-related features. The project uses **Gradient Boosting Regression** to generate accurate predictions and is deployed as an interactive **Flask web application** for real-time forecasting.

---

## 📌 Overview

Bike-sharing systems generate large amounts of usage data that can be leveraged to optimize operations and improve resource allocation. This project applies Machine Learning techniques to forecast bike rental demand by analyzing historical weather and temporal data.

The model was trained and evaluated on the **Kaggle Bike Sharing Demand dataset**, achieving excellent predictive performance.

---

## 🚀 Features

* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data preprocessing and feature engineering
* 🤖 Gradient Boosting Regression model training
* 📈 Model evaluation and comparison with baseline algorithms
* 🌐 Flask-based web application for real-time predictions
* ⚡ Instant demand forecasting through a user-friendly interface

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Flask
* Joblib

---

## 📂 Project Structure

```text
Bike-Sharing-Demand-Prediction/
│
├── dataset/
├── model/
├── static/
├── templates/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Serialization using Joblib
8. Flask Web Application Deployment

---

## 📈 Model Performance

### 🏆 Best Model: Gradient Boosting Regressor

* **R² Score:** **0.9391**
* Outperformed multiple baseline models including:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor

The final model was selected based on superior predictive accuracy and generalization performance.

---

## 💡 Key Highlights

* Built a complete end-to-end Machine Learning pipeline.
* Achieved an **R² score of 0.9391** using Gradient Boosting.
* Performed feature engineering using weather and temporal variables.
* Deployed the trained model as a **Flask web application** for real-time demand prediction.
* Demonstrates practical experience in Machine Learning model development and deployment.

---

## ▶️ Getting Started

### Clone the repository

```bash
git clone https://github.com/Maariakh-cs/Bike-Sharing-Demand-Prediction.git
```

### Navigate to the project directory

```bash
cd Bike-Sharing-Demand-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Flask application

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📊 Dataset

This project uses the **Bike Sharing Demand** dataset from Kaggle, containing historical information on bike rentals along with weather and time-related attributes used for demand forecasting.

---

## 🔮 Future Improvements

* Hyperparameter optimization
* Integration with XGBoost and LightGBM models
* Cloud deployment (Render/AWS/Azure)
* Interactive dashboard for demand visualization
* REST API for external integrations

---

## 👩‍💻 Author

**Maaria Khan**

* GitHub: https://github.com/Maariakh-cs
* LinkedIn: https://www.linkedin.com/in/maariakh-cs/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
