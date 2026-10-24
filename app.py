
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from html import escape

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_PATH = ROOT / "models" / "final_pipeline.pkl"
REPORTS_PATH = ROOT / "reports"

st.set_page_config(
    page_title="Retail Sales & Profit Intelligence",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 2. CSS - TEXT VISIBILITY FIX
# ============================================================

st.markdown("""
<style>

/* Main app headings - LIGHT TEXT */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] h1 *,
[data-testid="stMain"] h2 *,
[data-testid="stMain"] h3 * {
    color: #F9FAFB !important;
    -webkit-text-fill-color: #F9FAFB !important;
    opacity: 1 !important;
}

/* Main app paragraphs */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {
    color: #D1D5DB !important;
    -webkit-text-fill-color: #D1D5DB !important;
    opacity: 1 !important;
}

/* Captions */
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: #CBD5E1 !important;
    -webkit-text-fill-color: #CBD5E1 !important;
}

/* Widget labels */
[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] label {
    color: #E5E7EB !important;
    -webkit-text-fill-color: #E5E7EB !important;
}

/* Slider labels */
[data-testid="stMain"] [data-testid="stSlider"] label,
[data-testid="stMain"] [data-testid="stSlider"] label * {
    color: #E5E7EB !important;
    -webkit-text-fill-color: #E5E7EB !important;
}

/* Selectbox, date input and text input labels */
[data-testid="stMain"] [data-testid="stSelectbox"] label,
[data-testid="stMain"] [data-testid="stMultiSelect"] label,
[data-testid="stMain"] [data-testid="stDateInput"] label,
[data-testid="stMain"] [data-testid="stNumberInput"] label,
[data-testid="stMain"] [data-testid="stTextInput"] label {
    color: #E5E7EB !important;
    -webkit-text-fill-color: #E5E7EB !important;
}

/* Metric cards */
.ri-metric-card {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    min-height: 88px !important;
    margin-bottom: 8px !important;
    box-sizing: border-box !important;
}

.ri-metric-card .ri-metric-label {
    color: #374151 !important;
    -webkit-text-fill-color: #374151 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

.ri-metric-card .ri-metric-value {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    margin-top: 6px !important;
    opacity: 1 !important;
}

/* Native Streamlit metrics */
[data-testid="stMain"] [data-testid="stMetricLabel"],
[data-testid="stMain"] [data-testid="stMetricLabel"] *,
[data-testid="stMain"] [data-testid="stMetricValue"],
[data-testid="stMain"] [data-testid="stMetricValue"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #101827 !important;
}

section[data-testid="stSidebar"] * {
    color: #F9FAFB !important;
}

/* Sidebar radio */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #F9FAFB !important;
}

/* Tabs */
[data-testid="stMain"] button[data-baseweb="tab"] {
    color: #E5E7EB !important;
}

/* Expander headings */
[data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stMain"] [data-testid="stExpander"] summary p {
    color: #E5E7EB !important;
}

/* Do not force a global background color.
   Do not modify dataframe colors. */

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    required = [
        "Order_Date", "Order_ID", "Customer_Gender",
        "Customer_Age", "City", "Region",
        "Product_Category", "Product_Name", "Brand",
        "Quantity", "Unit_Price", "Discount",
        "Shipping_Cost", "Payment_Method",
        "Delivery_Days", "Salesperson",
        "Total_Sale", "Profit", "Returned",
        "Customer_Rating"
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    data["Order_Date"] = pd.to_datetime(
        data["Order_Date"], errors="coerce"
    )

    data = data.dropna(subset=["Order_Date"]).copy()
    data = data.drop_duplicates().copy()

    data["Order_Year"] = data["Order_Date"].dt.year
    data["Order_Month"] = data["Order_Date"].dt.month
    data["Month_Name"] = data["Order_Date"].dt.strftime("%b")
    data["Quarter"] = "Q" + data["Order_Date"].dt.quarter.astype(str)

    data["Age_Group"] = pd.cut(
        data["Customer_Age"],
        bins=[0, 24, 34, 44, 54, 120],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
        include_lowest=True
    ).astype("object")

    data["Profit_Margin"] = np.where(
        data["Total_Sale"] > 0,
        data["Profit"] / data["Total_Sale"] * 100,
        0
    )

    data["Discount_Amount"] = (
        data["Quantity"] * data["Unit_Price"] * data["Discount"]
    )

    data["Revenue_Before_Discount"] = (
        data["Quantity"] * data["Unit_Price"]
    )

    data["Delivery_Status"] = np.where(
        data["Delivery_Days"] > 7,
        "Late",
        "On Time"
    )

    return data


@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        return None
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Please make sure data/dataset.csv exists.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    MODEL_ERROR = ""
except Exception as error:
    model = None
    MODEL_ERROR = str(error)


# ============================================================
# 4. COMMON FUNCTIONS
# ============================================================

def money(value):
    return f"PKR {value:,.0f}"


def metric_card(label, value):
    """
    Important:
    HTML is deliberately assembled without indentation.
    This prevents Streamlit Markdown from displaying it as code.
    """

    safe_label = escape(str(label))
    safe_value = escape(str(value))

    html = (
        '<div class="ri-metric-card">'
        '<div class="ri-metric-label">'
        + safe_label +
        '</div>'
        '<div class="ri-metric-value">'
        + safe_value +
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def show_empty_message():
    st.warning(
        "No records match the selected filters. "
        "Please change your filters."
    )


def filters(data, key_prefix="main"):

    filtered = data.copy()

    with st.sidebar.expander(
        "Advanced Filters",
        expanded=True
    ):

        regions_all = sorted(
            data["Region"].dropna().unique().tolist()
        )

        categories_all = sorted(
            data["Product_Category"].dropna().unique().tolist()
        )

        products_all = sorted(
            data["Product_Name"].dropna().unique().tolist()
        )

        regions = st.multiselect(
            "Region",
            regions_all,
            default=regions_all,
            key=f"{key_prefix}_regions"
        )

        categories = st.multiselect(
            "Category",
            categories_all,
            default=categories_all,
            key=f"{key_prefix}_categories"
        )

        products = st.multiselect(
            "Product",
            products_all,
            default=[],
            key=f"{key_prefix}_products"
        )

        min_date = data["Order_Date"].min().date()
        max_date = data["Order_Date"].max().date()

        selected_dates = st.date_input(
            "Order Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_dates"
        )

    filtered = filtered[
        filtered["Region"].isin(regions)
        & filtered["Product_Category"].isin(categories)
    ]

    if products:
        filtered = filtered[
            filtered["Product_Name"].isin(products)
        ]

    if isinstance(selected_dates, (tuple, list)):

        if len(selected_dates) == 2:
            filtered = filtered[
                filtered["Order_Date"].dt.date.between(
                    selected_dates[0],
                    selected_dates[1]
                )
            ]

        elif len(selected_dates) == 1:
            filtered = filtered[
                filtered["Order_Date"].dt.date == selected_dates[0]
            ]

    return filtered.copy()


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Retail Intelligence")

pages = [
    "Home",
    "Dataset Explorer",
    "Sales Dashboard",
    "Product & Regional Analysis",
    "Customer & Delivery Analysis",
    "Model Evaluation",
    "Profit Prediction",
    "Business Recommendations"
]

page = st.sidebar.radio(
    "Navigate",
    pages
)

st.sidebar.caption("Synthetic teaching dataset • PKR")


# ============================================================
# PAGE 1: HOME
# ============================================================

if page == "Home":

    st.title("Retail Sales Analytics & Profit Prediction")

    st.subheader(
        "Turning order data into practical business decisions"
    )

    st.write(
        "Business problem: A retail company needs to understand "
        "revenue, profitability, returns, delivery experience and "
        "salesperson performance, then estimate profit for a proposed order."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Orders in dataset", f"{len(df):,}")

    with col2:
        metric_card(
            "Product categories",
            df["Product_Category"].nunique()
        )

    with col3:
        metric_card(
            "Date coverage",
            f"{df['Order_Date'].min():%b %Y} - "
            f"{df['Order_Date'].max():%b %Y}"
        )

    st.subheader("Project Objectives")

    st.markdown("""
    - Discover profitable products, categories and regions.
    - Understand discount, delivery and return patterns.
    - Estimate profit for proposed orders using machine learning.
    - Provide an interactive dashboard for business users.
    """)

    st.info(
        "Dataset note: If this dataset is synthetic, its results are "
        "demonstrations and not claims about a real retailer."
    )

    with st.expander("Dataset Description"):
        st.write(
            "Each row represents a retail order. It contains customer, "
            "product, order, delivery, sales and profit attributes."
        )

    st.caption(
        "Student: Add your name here | Course: Data Science and Machine Learning"
    )


# ============================================================
# PAGE 2: DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    f = filters(df, "explorer")

    nrows = st.slider(
        "Rows to Preview",
        min_value=5,
        max_value=500,
        value=100
    )

    metric_card("Filtered rows", f"{len(f):,}")

    st.dataframe(
        f.head(nrows),
        use_container_width=True
    )

    st.download_button(
        "Download Filtered CSV",
        data=f.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )

    tab1, tab2, tab3 = st.tabs([
        "Data Types",
        "Missing Values",
        "Summary Statistics"
    ])

    with tab1:
        st.dataframe(
            pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values
            }),
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            df.isna().sum().rename("Missing Count").to_frame(),
            use_container_width=True
        )

    with tab3:
        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )


# ============================================================
# PAGE 3: SALES DASHBOARD
# ============================================================

elif page == "Sales Dashboard":

    st.title("Sales Dashboard")

    f = filters(df, "dashboard")

    st.subheader("Dashboard Date Filter")

    all_min = df["Order_Date"].min().date()
    all_max = df["Order_Date"].max().date()

    default_min = (
        f["Order_Date"].min().date()
        if not f.empty else all_min
    )

    default_max = (
        f["Order_Date"].max().date()
        if not f.empty else all_max
    )

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_min,
            min_value=all_min,
            max_value=all_max,
            key="dashboard_start"
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_max,
            min_value=all_min,
            max_value=all_max,
            key="dashboard_end"
        )

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    f = f[
        f["Order_Date"].dt.date.between(
            start_date,
            end_date
        )
    ].copy()

    st.divider()

    if f.empty:

        show_empty_message()

    else:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Total Sales", money(f["Total_Sale"].sum()))

        with col2:
            metric_card("Total Profit", money(f["Profit"].sum()))

        with col3:
            metric_card("Total Orders", f"{len(f):,}")

        with col4:
            metric_card(
                "Average Order Value",
                money(f["Total_Sale"].mean())
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "Average Profit",
                money(f["Profit"].mean())
            )

        with col2:
            metric_card(
                "Return Rate",
                f"{f['Returned'].eq('Yes').mean() * 100:.1f}%"
            )

        with col3:
            metric_card(
                "Average Rating",
                f"{f['Customer_Rating'].mean():.2f}/5"
            )

        with col4:
            metric_card(
                "Average Delivery Days",
                f"{f['Delivery_Days'].mean():.1f}"
            )

        monthly = (
            f.assign(
                Month=f["Order_Date"].dt.to_period("M").astype(str)
            )
            .groupby("Month", as_index=False)
            .agg(
                Sales=("Total_Sale", "sum"),
                Profit=("Profit", "sum")
            )
        )

        st.plotly_chart(
            px.line(
                monthly,
                x="Month",
                y=["Sales", "Profit"],
                markers=True,
                title="Monthly Sales and Profit"
            ),
            use_container_width=True
        )

        st.plotly_chart(
            px.histogram(
                f,
                x="Profit",
                nbins=35,
                title="Order Profit Distribution"
            ),
            use_container_width=True
        )


# ============================================================
# PAGE 4: PRODUCT & REGIONAL ANALYSIS
# ============================================================

elif page == "Product & Regional Analysis":

    st.title("Product & Regional Analysis")

    f = filters(df, "product_region")

    if f.empty:

        show_empty_message()

    else:

        left, right = st.columns(2)

        category_sales = (
            f.groupby("Product_Category", as_index=False)
            ["Total_Sale"].sum()
        )

        category_profit = (
            f.groupby("Product_Category", as_index=False)
            ["Profit"].sum()
        )

        with left:
            st.plotly_chart(
                px.bar(
                    category_sales,
                    x="Product_Category",
                    y="Total_Sale",
                    title="Sales by Category"
                ),
                use_container_width=True
            )

        with right:
            st.plotly_chart(
                px.bar(
                    category_profit,
                    x="Product_Category",
                    y="Profit",
                    title="Profit by Category"
                ),
                use_container_width=True
            )

        top_products = (
            f.groupby("Product_Name", as_index=False)
            .agg(
                Sales=("Total_Sale", "sum"),
                Profit=("Profit", "sum")
            )
            .sort_values("Sales", ascending=False)
            .head(10)
        )

        st.plotly_chart(
            px.bar(
                top_products,
                x="Sales",
                y="Product_Name",
                orientation="h",
                title="Top 10 Products by Sales",
                hover_data=["Profit"]
            ),
            use_container_width=True
        )

        brand_sales = (
            f.groupby("Brand", as_index=False)
            ["Total_Sale"].sum()
            .sort_values("Total_Sale", ascending=False)
        )

        region_profit = (
            f.groupby("Region", as_index=False)
            .agg(
                Sales=("Total_Sale", "sum"),
                Profit=("Profit", "sum")
            )
        )

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                px.bar(
                    brand_sales,
                    x="Brand",
                    y="Total_Sale",
                    title="Revenue by Brand"
                ),
                use_container_width=True
            )

        with right:
            st.plotly_chart(
                px.bar(
                    region_profit,
                    x="Region",
                    y="Profit",
                    title="Profit by Region"
                ),
                use_container_width=True
            )

        salesperson = (
            f.groupby("Salesperson", as_index=False)
            .agg(
                Sales=("Total_Sale", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order_ID", "count")
            )
            .sort_values("Profit", ascending=False)
        )

        st.subheader("Salesperson Performance")

        st.dataframe(
            salesperson,
            use_container_width=True
        )


# ============================================================
# PAGE 5: CUSTOMER & DELIVERY ANALYSIS
# ============================================================

elif page == "Customer & Delivery Analysis":

    st.title("Customer & Delivery Analysis")

    f = filters(df, "customer_delivery")

    if f.empty:

        show_empty_message()

    else:

        left, right = st.columns(2)

        gender_sales = (
            f.groupby("Customer_Gender", as_index=False)
            ["Total_Sale"].sum()
        )

        age_sales = (
            f.groupby(
                "Age_Group",
                as_index=False,
                observed=False
            )["Total_Sale"].sum()
        )

        with left:
            st.plotly_chart(
                px.bar(
                    gender_sales,
                    x="Customer_Gender",
                    y="Total_Sale",
                    title="Sales by Gender"
                ),
                use_container_width=True
            )

        with right:
            st.plotly_chart(
                px.bar(
                    age_sales,
                    x="Age_Group",
                    y="Total_Sale",
                    title="Sales by Age Group"
                ),
                use_container_width=True
            )

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                px.histogram(
                    f,
                    x="Customer_Rating",
                    nbins=5,
                    title="Customer Rating Distribution"
                ),
                use_container_width=True
            )

        with right:
            st.plotly_chart(
                px.scatter(
                    f,
                    x="Delivery_Days",
                    y="Customer_Rating",
                    color="Returned",
                    title="Delivery Days vs Customer Rating"
                ),
                use_container_width=True
            )

        return_rate = (
            f.assign(
                Return_Flag=f["Returned"].eq("Yes").astype(int)
            )
            .groupby("Product_Category", as_index=False)
            .agg(Return_Rate=("Return_Flag", "mean"))
        )

        return_rate["Return_Rate"] *= 100

        st.plotly_chart(
            px.bar(
                return_rate,
                x="Product_Category",
                y="Return_Rate",
                title="Return Rate by Category (%)"
            ),
            use_container_width=True
        )

        late = f[f["Delivery_Days"] > 7]
        on_time = f[f["Delivery_Days"] <= 7]

        late_rate = (
            late["Returned"].eq("Yes").mean() * 100
            if not late.empty else 0
        )

        on_time_rate = (
            on_time["Returned"].eq("Yes").mean() * 100
            if not on_time.empty else 0
        )

        col1, col2 = st.columns(2)

        with col1:
            metric_card(
                "Late Delivery Return Rate",
                f"{late_rate:.1f}%"
            )

        with col2:
            metric_card(
                "On-time Delivery Return Rate",
                f"{on_time_rate:.1f}%"
            )

        st.plotly_chart(
            px.pie(
                f,
                names="Payment_Method",
                title="Payment Method Distribution"
            ),
            use_container_width=True
        )


# ============================================================
# PAGE 6: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    metrics_path = REPORTS_PATH / "model_metrics.csv"
    predictions_path = REPORTS_PATH / "test_predictions.csv"
    importance_path = REPORTS_PATH / "permutation_importance.csv"

    if metrics_path.exists():

        metrics = pd.read_csv(metrics_path)

        st.subheader("Model Comparison")

        st.dataframe(
            metrics,
            use_container_width=True
        )

        if {"Model", "RMSE"}.issubset(metrics.columns):

            st.plotly_chart(
                px.bar(
                    metrics,
                    x="Model",
                    y="RMSE",
                    title="Model Comparison - Lower RMSE is Better"
                ),
                use_container_width=True
            )

    else:

        st.warning(
            "model_metrics.csv was not found in the reports folder."
        )

    if predictions_path.exists():

        predictions = pd.read_csv(predictions_path)

        required = {
            "Actual_Profit",
            "Predicted_Profit"
        }

        if required.issubset(predictions.columns):

            valid = predictions[
                ["Actual_Profit", "Predicted_Profit"]
            ].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            if not valid.empty:

                st.subheader("Actual vs Predicted Profit")

                fig = px.scatter(
                    valid,
                    x="Actual_Profit",
                    y="Predicted_Profit",
                    title="Actual vs Predicted Profit"
                )

                low = min(
                    valid["Actual_Profit"].min(),
                    valid["Predicted_Profit"].min()
                )

                high = max(
                    valid["Actual_Profit"].max(),
                    valid["Predicted_Profit"].max()
                )

                fig.add_trace(
                    go.Scatter(
                        x=[low, high],
                        y=[low, high],
                        mode="lines",
                        name="Ideal Prediction"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                residuals = (
                    valid["Actual_Profit"]
                    - valid["Predicted_Profit"]
                )

                st.plotly_chart(
                    px.histogram(
                        x=residuals,
                        nbins=35,
                        title="Residual Distribution",
                        labels={"x": "Actual - Predicted"}
                    ),
                    use_container_width=True
                )

            else:

                st.warning(
                    "No valid numeric prediction rows were found."
                )

        else:

            st.warning(
                "Prediction file must contain Actual_Profit "
                "and Predicted_Profit columns."
            )

    else:

        st.warning(
            "test_predictions.csv was not found."
        )

    if importance_path.exists():

        importance = pd.read_csv(importance_path)

        if (
            {"Feature", "Importance"}.issubset(importance.columns)
            and not importance.empty
        ):

            importance = (
                importance.head(12)
                .sort_values("Importance")
            )

            st.subheader("Model Explainability")

            st.plotly_chart(
                px.bar(
                    importance,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Permutation Feature Importance"
                ),
                use_container_width=True
            )

            st.caption(
                "Permutation importance measures how model performance "
                "changes when a feature is shuffled. It does not prove "
                "that a feature causes profit changes."
            )

    else:

        st.info(
            "permutation_importance.csv was not found."
        )


# ============================================================
# PAGE 7: PROFIT PREDICTION
# ============================================================

elif page == "Profit Prediction":

    st.title("Profit Prediction")

    st.write(
        "Enter the proposed order details to estimate expected profit."
    )

    if model is None:

        st.error("Prediction model could not be loaded.")

        st.write(f"Expected file: {MODEL_PATH}")

        if MODEL_ERROR:
            st.code(MODEL_ERROR)

        st.stop()

    gender_values = sorted(
        df["Customer_Gender"].dropna().astype(str).unique()
    )

    city_values = sorted(
        df["City"].dropna().astype(str).unique()
    )

    region_values = sorted(
        df["Region"].dropna().astype(str).unique()
    )

    category_values = sorted(
        df["Product_Category"].dropna().astype(str).unique()
    )

    payment_values = sorted(
        df["Payment_Method"].dropna().astype(str).unique()
    )

    salesperson_values = sorted(
        df["Salesperson"].dropna().astype(str).unique()
    )

    if not gender_values:
        gender_values = ["Other"]

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        gender = col1.selectbox(
            "Customer Gender",
            gender_values
        )

        age = col2.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

        city = col3.selectbox(
            "City",
            city_values
        )

        region = col1.selectbox(
            "Region",
            region_values
        )

        category = col2.selectbox(
            "Product Category",
            category_values
        )

        products = sorted(
            df.loc[
                df["Product_Category"].astype(str) == category,
                "Product_Name"
            ].dropna().astype(str).unique()
        )

        product = col3.selectbox(
            "Product",
            products
        )

        brands = sorted(
            df.loc[
                df["Product_Name"].astype(str) == product,
                "Brand"
            ].dropna().astype(str).unique()
        )

        brand = col1.selectbox(
            "Brand",
            brands if brands else ["Unknown"]
        )

        quantity = col2.number_input(
            "Quantity",
            min_value=1,
            max_value=100,
            value=2
        )

        price = col3.number_input(
            "Unit Price (PKR)",
            min_value=1.0,
            value=2500.0,
            step=100.0
        )

        discount = col1.slider(
            "Discount",
            min_value=0.0,
            max_value=0.60,
            value=0.10,
            step=0.01
        )

        shipping = col2.number_input(
            "Shipping Cost (PKR)",
            min_value=0.0,
            max_value=10000.0,
            value=350.0,
            step=50.0
        )

        payment = col3.selectbox(
            "Payment Method",
            payment_values
        )

        delivery = col1.slider(
            "Expected Delivery Days",
            min_value=1,
            max_value=30,
            value=4
        )

        salesperson = col2.selectbox(
            "Salesperson",
            salesperson_values
        )

        order_date = col3.date_input(
            "Expected Order Date"
        )

        submitted = st.form_submit_button(
            "Predict Expected Profit",
            type="primary"
        )

    low_boundary = st.number_input(
        "Loss / Low Profit Boundary (PKR)",
        value=0.0,
        step=100.0
    )

    medium_boundary = st.number_input(
        "Low / Medium Profit Boundary (PKR)",
        value=1500.0,
        step=100.0
    )

    high_boundary = st.number_input(
        "Medium / High Profit Boundary (PKR)",
        value=3000.0,
        step=100.0
    )

    if submitted:

        if not (
            low_boundary < medium_boundary < high_boundary
        ):

            st.error(
                "Profit boundaries must be in increasing order."
            )

        else:

            prediction_row = pd.DataFrame([{
                "Customer_Gender": gender,
                "Customer_Age": age,
                "City": city,
                "Region": region,
                "Product_Category": category,
                "Product_Name": product,
                "Brand": brand,
                "Quantity": quantity,
                "Unit_Price": price,
                "Discount": discount,
                "Shipping_Cost": shipping,
                "Payment_Method": payment,
                "Delivery_Days": delivery,
                "Salesperson": salesperson,
                "Order_Year": order_date.year,
                "Order_Month": order_date.month,
                "Quarter": (order_date.month - 1) // 3 + 1
            }])

            try:

                prediction = float(
                    model.predict(prediction_row)[0]
                )

                if prediction < low_boundary:
                    label = "Loss"

                elif prediction < medium_boundary:
                    label = "Low Profit"

                elif prediction < high_boundary:
                    label = "Medium Profit"

                else:
                    label = "High Profit"

                metric_card(
                    "Predicted Profit",
                    money(prediction)
                )

                st.subheader(
                    f"Estimated Category: {label}"
                )

                if prediction < 0:
                    st.warning(
                        "This order is predicted to make a loss. "
                        "Review price, discount and shipping costs."
                    )

                result = prediction_row.copy()
                result["Predicted_Profit"] = prediction
                result["Profit_Category"] = label

                st.download_button(
                    "Download Prediction Result",
                    data=result.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="profit_prediction.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error(
                    "Prediction failed. The model input columns "
                    "or category values may not match."
                )

                st.exception(error)


# ============================================================
# PAGE 8: BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":

    st.title("Business Recommendations")

    category_summary = (
        df.groupby("Product_Category")
        .agg(
            Sales=("Total_Sale", "sum"),
            Profit=("Profit", "sum"),
            Return_Rate=(
                "Returned",
                lambda x: x.eq("Yes").mean() * 100
            )
        )
        .reset_index()
    )

    if category_summary.empty:

        st.warning(
            "Not enough data to generate recommendations."
        )

    else:

        best_category = category_summary.loc[
            category_summary["Profit"].idxmax()
        ]

        highest_returns = category_summary.loc[
            category_summary["Return_Rate"].idxmax()
        ]

        late_ratings = df.loc[
            df["Delivery_Days"] > 7,
            "Customer_Rating"
        ].dropna()

        on_time_ratings = df.loc[
            df["Delivery_Days"] <= 7,
            "Customer_Rating"
        ].dropna()

        late_average = (
            late_ratings.mean()
            if not late_ratings.empty else np.nan
        )

        on_time_average = (
            on_time_ratings.mean()
            if not on_time_ratings.empty else np.nan
        )

        discount_corr = df[
            ["Discount", "Profit"]
        ].corr().iloc[0, 1]

        top_salesperson = (
            df.groupby("Salesperson")["Profit"]
            .sum()
            .idxmax()
        )

        late_text = (
            f"{late_average:.2f}"
            if pd.notna(late_average) else "not available"
        )

        on_time_text = (
            f"{on_time_average:.2f}"
            if pd.notna(on_time_average) else "not available"
        )

        corr_text = (
            f"{discount_corr:.2f}"
            if pd.notna(discount_corr) else "not available"
        )

        recommendations = [

            (
                "Category Profitability",
                f"{best_category['Product_Category']} has the highest "
                f"total profit ({money(best_category['Profit'])}).",
                "Evidence: Total profit grouped by product category.",
                "Protect product availability and test carefully "
                "measured promotions.",
                "Better stock allocation and profit stability."
            ),

            (
                "Product Returns",
                f"{highest_returns['Product_Category']} has the highest "
                f"observed return rate "
                f"({highest_returns['Return_Rate']:.1f}%).",
                "Evidence: Return rate grouped by category.",
                "Review product descriptions, packaging and quality checks.",
                "Fewer avoidable returns and lower return costs."
            ),

            (
                "Discount Management",
                f"Discount-profit correlation is {corr_text}. "
                "Correlation does not prove causation.",
                "Evidence: Correlation between Discount and Profit.",
                "Set discount limits and measure contribution profit "
                "before and after promotions.",
                "Reduced margin erosion."
            ),

            (
                "Delivery Experience",
                f"Average rating is {on_time_text} for orders delivered "
                f"within 7 days and {late_text} for later orders.",
                "Evidence: Average customer rating by delivery group.",
                "Monitor late-delivery routes and notify customers "
                "about delivery delays.",
                "Better customer experience."
            ),

            (
                "Salesperson Performance",
                f"{top_salesperson} has the highest total profit "
                "in this dataset.",
                "Evidence: Total profit grouped by salesperson.",
                "Share effective selling practices and track profit "
                "per order, not only sales volume.",
                "More balanced sales performance."
            )
        ]

        for title, finding, evidence, action, result in recommendations:

            with st.expander(title, expanded=True):

                st.markdown(
                    f"**Finding:** {finding}\n\n"
                    f"**Evidence:** {evidence}\n\n"
                    f"**Recommendation:** {action}\n\n"
                    f"**Expected Result:** {result}"
                )

    st.caption(
        "Recommendations are calculated from the included dataset "
        "and should be validated against real business data."
    )


# ============================================================
# END OF APPLICATION
# ============================================================
