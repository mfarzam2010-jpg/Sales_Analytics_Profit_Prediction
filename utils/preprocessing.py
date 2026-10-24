import numpy as np
import pandas as pd
def engineer_features(df):
    d=df.copy()
    d['Order_Date']=pd.to_datetime(d['Order_Date'],errors='coerce')
    d['Order_Year']=d.Order_Date.dt.year
    d['Order_Month']=d.Order_Date.dt.month
    d['Month_Name']=d.Order_Date.dt.month_name()
    d['Quarter']=d.Order_Date.dt.quarter
    d['Day_Name']=d.Order_Date.dt.day_name()
    d['Age_Group']=pd.cut(d.Customer_Age,[0,24,34,44,54,100],labels=['18-24','25-34','35-44','45-54','55+']).astype(str)
    d['Profit_Margin']=np.where(d.Total_Sale>0,d.Profit/d.Total_Sale*100,0)
    d['Discount_Amount']=d.Quantity*d.Unit_Price*d.Discount
    d['Revenue_Before_Discount']=d.Quantity*d.Unit_Price
    d['Delivery_Status']=np.where(d.Delivery_Days>7,'Late','On Time')
    d['Order_Size_Category']=pd.cut(d.Quantity,[0,1,3,6,100],labels=['Single','Small','Medium','Large']).astype(str)
    return d
