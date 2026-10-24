import plotly.express as px
def category_sales(df):
    x=df.groupby('Product_Category',as_index=False).Total_Sale.sum()
    return px.bar(x,x='Product_Category',y='Total_Sale',title='Sales by category')
