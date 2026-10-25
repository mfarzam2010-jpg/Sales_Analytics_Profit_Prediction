# Retail Sales Analytics & Profit Prediction

A data analytics and machine learning project that analyzes retail sales performance and predicts business profit using a trained Random Forest model. The project includes an interactive Streamlit dashboard with business insights and recommendations.

## 1. Business Problem

Retail businesses need to understand sales performance, identify profitable products and regions, and estimate the profit of future orders.

This project uses data analytics and machine learning to support data-driven business decisions.

## 2. Project Objectives

- Analyze sales and profit trends.
- Identify profitable product categories and regions.
- Understand customer and delivery performance.
- Build a machine learning model to predict profit.
- Provide actionable business recommendations.
- Develop an interactive Streamlit application.

## 3. Dataset Description

- **Dataset:** `data/dataset.csv`
- **Records:** 2,401 orders
- **Type:** Synthetic educational dataset
- **Target Variable:** Profit

The dataset contains order, product, customer, regional, discount, delivery, and sales information.

*Note: This is a synthetic educational dataset, not actual company transaction data.*

## 4. Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub
- Docker
- Pytest

## 5. Data Cleaning and Preprocessing

- Date conversion and formatting.
- Duplicate record checks and removal.
- Numeric data type conversion.
- Missing value checks.
- Handling missing values through preprocessing.
- Investigation of unusual values.
- Preparation of features for machine learning.

## 6. Exploratory Data Analysis (EDA)

The analysis covers:

- Sales and profit trends.
- Product and category performance.
- Regional performance.
- Customer analysis.
- Discount and return analysis.
- Delivery performance.
- Payment methods.
- Salesperson performance.

## 7. Feature Engineering

Relevant numerical and categorical features are prepared for machine learning.

A preprocessing pipeline prepares input data before model prediction.

## 8. Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## 9. Model Evaluation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 1,417.59 | 2,443.81 | 0.8924 |
| Decision Tree | 1,977.61 | 3,263.48 | 0.8082 |
| Linear Regression | 2,921.48 | 4,397.18 | 0.6518 |

## 10. Best Model

Random Forest Regressor achieved the best evaluation results among the tested models.

Saved pipeline:

`models/final_pipeline.pkl`

The saved pipeline includes preprocessing and the trained prediction model.

## 11. Streamlit Application Features

The application contains eight sections:

1. Home
2. Dataset Explorer
3. Sales Dashboard
4. Product & Regional Analysis
5. Customer & Delivery Analysis
6. Model Evaluation
7. Profit Prediction
8. Business Recommendations

Users can explore data, interact with charts, apply filters, and generate profit predictions.

## 12. Business Recommendations

The application provides insights and recommendations related to:

- Category profitability.
- Product returns.
- Discount management.
- Delivery performance.
- Salesperson performance.

## 13. Project Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/mfarzam2010-jpg/Sales_Analytics_Profit_Prediction.git
```

### Step 2: Open the project folder

```bash
cd Sales_Analytics_Profit_Prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate.bat
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

## 14. Live Application

**Streamlit Deployment:**

https://salesanalyticsprofitprediction-zuqkjcug4s3dxsw9w8ams9.streamlit.app/

## 15. Application Screenshots

### Screenshot 1
<img src="./screenshots/aa.png" width="800" />

### Screenshot 2
<img src="./screenshots/image(1).png" width="800" />

### Screenshot 3
<img src="./screenshots/image(10).png" width="800" />

### Screenshot 4
<img src="./screenshots/image(2).png" width="800" />

### Screenshot 5
<img src="./screenshots/image(20261025-085656).png" width="800" />

### Screenshot 6
<img src="./screenshots/image(20261025-085701).png" width="800" />

### Screenshot 7
<img src="./screenshots/image(20261025-085706).png" width="800" />

### Screenshot 8
<img src="./screenshots/image(20261025-085727).png" width="800" />

### Screenshot 9
<img src="./screenshots/image(20261025-085844).png" width="800" />

### Screenshot 10
<img src="./screenshots/image(20261025-085851).png" width="800" />

### Screenshot 11
<img src="./screenshots/image(3).png" width="800" />

### Screenshot 12
<img src="./screenshots/image(4).png" width="800" />

### Screenshot 13
<img src="./screenshots/image(5).png" width="800" />

### Screenshot 14
<img src="./screenshots/image(6).png" width="800" />

### Screenshot 15
<img src="./screenshots/image(7).png" width="800" />

### Screenshot 16
<img src="./screenshots/image(8).png" width="800" />

### Screenshot 17
<img src="./screenshots/image(9).png" width="800" />

### Screenshot 18
<img src="./screenshots/profit_prediction_screenshot_1.png" width="800" />

### Screenshot 19
<img src="./screenshots/profit_prediction_screenshot_2.png" width="800" />

### Screenshot 20
<img src="./screenshots/z.png" width="800" />

### Screenshot 21
<img src="./screenshots/zz.png" width="800" />

### Screenshot 22
<img src="./screenshots/zzz.png" width="800" />

### Screenshot 23
<img src="./screenshots/zzzz.png" width="800" />

## 16. GitHub Repository

https://github.com/mfarzam2010-jpg/Sales_Analytics_Profit_Prediction

## 17. Project Structure

```text
Sales_Analytics_Profit_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── model.pkl
│   └── final_pipeline.pkl
│
├── notebooks/
│   └── project_analysis.ipynb
│
├── pages/
│   └── README.txt
│
├── reports/
│   ├── business_insights.txt
│   ├── model_metrics.csv
│   ├── permutation_importance.csv
│   ├── project_presentation.pptx
│   └── test_predictions.csv
│
├── screenshots/
│   └── Application screenshots
│
├── tests/
│   └── test_project.py
│
└── utils/
    ├── charts.py
    ├── data_loader.py
    ├── prediction.py
    └── preprocessing.py
```

## 18. Testing

The project includes automated tests for:

- Dataset availability and target column.
- Saved model availability.
- Successful model loading.
- Model prediction functionality.

Run the tests using:

```bash
python -m pytest -v
```

## 19. License

This project was developed for educational and academic purposes.