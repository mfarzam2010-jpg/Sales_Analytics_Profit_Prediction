# Retail Sales Analytics & Profit Prediction

## Business problem
A retail business needs to understand sales, profit, products, regions, discounts, returns, delivery experience and salesperson performance, and estimate profit for a proposed order.

## Dataset description
`data/dataset.csv` contains 2,401 generated order rows with the assignment's requested fields. **Important:** this is a synthetic teaching dataset created because no real company CSV was supplied. It is suitable for demonstrating the complete workflow, but its findings and model scores must not be presented as real-company evidence. Replace it with an authorized real dataset if your instructor requires original business data.

## Project objectives
- Clean and inspect retail order data.
- Engineer date, age, margin, discount and delivery features.
- Explore sales, profit, returns, discounts and customer experience.
- Train and compare Linear Regression, Decision Tree and Random Forest.
- Predict expected profit through an interactive Streamlit form.

## Tools
Python, NumPy, Pandas, Matplotlib, Seaborn, Plotly, scikit-learn, Joblib, Streamlit, Jupyter, Docker and pytest.

## Data cleaning
- Parse `Order_Date` as datetime.
- Remove exact duplicate records.
- Convert numeric columns and inspect missing values.
- Impute missing values within model pipelines.
- Investigate unusual values rather than silently deleting them.
- Discount is stored as a decimal fraction (e.g. 0.10 = 10%).

## Feature engineering
Order year, month, quarter, month name, day name, age group, profit margin, discount amount, revenue before discount, delivery status and order-size category.

## EDA summary
The dashboard calculates sales, profit, order value, return rate, rating, delivery time, category/region/product/brand/salesperson performance, payment-method usage and discount/delivery relationships.

## Models
Linear Regression, Decision Tree Regressor and Random Forest Regressor. Model comparison is saved in `reports/model_metrics.csv`.

## Evaluation
MAE, MSE, RMSE and R² are calculated on a held-out test split. See `reports/model_metrics.csv`. The lowest-RMSE pipeline is saved as `models/final_pipeline.pkl`.

## Streamlit features
Eight sections: Home, Dataset Explorer, Sales Dashboard, Product & Regional Analysis, Customer & Delivery Analysis, Model Evaluation, Profit Prediction and Business Recommendations. Includes advanced filters, Plotly charts, configurable profit category boundaries, downloadable filtered data and prediction results, and error checks.

## Model explainability
The model comparison and actual-vs-predicted/residual views help assess performance. For stronger feature-level explainability, add SHAP or permutation importance as a future extension; do not interpret raw one-hot encoded coefficients as business causality.

## Business recommendations
See the Business Recommendations page. Each item contains Finding, Evidence, Recommendation and Expected Result. Validate all recommendations with real company data before acting.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker
```bash
docker build -t retail-profit-app .
docker run -p 8501:8501 retail-profit-app
```

## GitHub
1. Create a GitHub repository.
2. Upload this complete folder (not `.venv` or cache files).
3. Confirm `data/dataset.csv`, model files and `requirements.txt` are included.

## Streamlit Community Cloud
1. Sign in at https://share.streamlit.io/ using GitHub.
2. Choose this repository and branch.
3. Set main file path to `app.py`.
4. Deploy and test the public URL.
5. Add your deployment URL here.

**Deployment URL:** Add after deployment.

## Screenshots
Add real screenshots of your running app in `screenshots/` and link them here after testing.

## Submission checklist
- [ ] GitHub repository link
- [ ] Public Streamlit deployment link
- [ ] Jupyter notebook
- [ ] Final model and preprocessor pipeline
- [ ] README
- [ ] App screenshots
- [ ] Project presentation
- [ ] Business insight report
