# 🚲 Bike Sharing Demand Prediction using Machine Learning

An end-to-end Machine Learning project that predicts hourly bike rental demand based on weather conditions, seasonal patterns, and temporal features. The project implements a complete ML pipeline—from data preprocessing and feature engineering to model training, evaluation, and deployment using a Flask web application.

---

## 📌 Overview

Accurate demand forecasting helps bike-sharing providers optimize fleet management, reduce operational costs, and improve customer satisfaction.

This project uses historical bike rental data along with weather and time-related variables to predict hourly demand. Multiple regression algorithms were evaluated, with **Gradient Boosting Regression** selected as the best-performing model.

---

## 🚀 Features

- 📊 Comprehensive Exploratory Data Analysis (EDA)
- 🧹 Data preprocessing and feature engineering
- 🤖 Training and evaluation of multiple regression models
- 📈 Performance comparison using regression metrics
- 🏆 Gradient Boosting selected as the best-performing model
- 🌐 Flask web application for real-time demand prediction
- 💾 Model serialization for deployment

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- Joblib

---

## 📂 Project Structure

```text
Bike-Sharing-Demand-Prediction/
│
├── data/
├── notebooks/
├── models/
├── static/
├── templates/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Best Model Selection
9. Flask Deployment
10. Real-Time Prediction

---

## 🤖 Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

🏆 **Best Performing Model:** Gradient Boosting Regressor

---

## 📊 Model Performance

| Metric | Value |
|----------|-------|
| **R² Score** | **0.9391** |

The Gradient Boosting model demonstrated superior predictive performance and was selected for deployment.

---

## 💡 Key Highlights

- Built a complete end-to-end Machine Learning regression pipeline.
- Performed data preprocessing, feature engineering, and exploratory data analysis.
- Compared multiple regression algorithms to identify the best-performing model.
- Achieved an **R² Score of 0.9391** using **Gradient Boosting Regression**.
- Deployed the trained model as a **Flask web application** for real-time bike demand prediction.

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

### Run the application

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📚 Dataset

The project uses a bike-sharing dataset containing historical rental records and environmental factors such as:

- Season
- Weather conditions
- Temperature
- Humidity
- Wind speed
- Date and time attributes

These features are used to predict hourly bike rental demand.

---

## 🔮 Future Improvements

- Hyperparameter optimization
- XGBoost and LightGBM implementation
- Interactive dashboard for demand visualization
- REST API integration
- Cloud deployment using Render or AWS

---

## ⚠️ Disclaimer

This project was developed for educational and research purposes as part of a Machine Learning internship. The predictions are intended for demonstration and analytical use and should be validated before deployment in production environments.

---

## 👩‍💻 Author

**Maaria Khan**

- GitHub: https://github.com/Maariakh-cs
- LinkedIn: https://www.linkedin.com/in/maariakh-cs/

---

⭐ If you found this project useful, consider giving it a star on GitHub!
