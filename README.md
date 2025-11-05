# Predicting House Prices for Maple Valley Realty
This project aims to build a machine learning model that accurately predicts house prices based on key property and location attributes.
For Maple Valley Realty, the goal is to use data-driven insights to support property valuation, guide investment decisions, and enhance pricing transparency for clients.

## Objectives

* Analyze housing market trends to understand price variations.
* Identify key features influencing home values (e.g., location, area, condition).
* Build a predictive model that estimates property prices with high accuracy.
* Provide actionable insights to support real estate pricing strategy and buyer-seller decisions.

## Dataset Description

The dataset includes property characteristics such as:

* Number of bedrooms and bathrooms
* Living area (sqft) and lot size
* Year built and renovation year
* Location attributes (zipcode, neighborhood)
* House condition and view rating
* Sale price

Target Variable: Sale Price

## Data Preprocessing

Data cleaning and preparation steps:

1. Handled missing and null values.
2. Converted date and categorical variables to appropriate formats.
3. Encoded categorical variables using LabelEncoder and One-Hot Encoding.
4. Scaled numerical features using StandardScaler to improve model training.
5. Split dataset into training (80%) and testing (20%) sets.

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["Location"] = encoder.fit_transform(data["Location"])
data["PropertyType"] = encoder.fit_transform(data["PropertyType"])

# Scale numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols_to_scale = ["SizeInSqFt", "YearBuilt", "LotSize"]
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

# Split the data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
```

## Exploratory Data Analysis (EDA)

### Key Insights

* Living Area: Strong positive correlation with sale price.
* Location (Zipcode): Certain zip codes consistently have higher property values.
* Condition & Grade: Houses rated higher in condition and grade have significantly higher prices.
* Year Built: Newer houses command higher average prices.
* Renovation: Renovated properties tend to sell for more.

## Visualizations

* Heatmap of feature correlations with SalePrice.
* Distribution plots of house prices across neighborhoods.
* Scatterplots for key numeric predictors (e.g., living area vs. price).

## Model Building

The following models were trained and compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

The **Linear Regression** was best fitting for this model. 

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
coef = model.coef_
intercept = model.intercept_
print(f"Coefficients: {coef}")
print(f"Intercept: {intercept}")
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {np.sqrt(mse)}")
print(f"R2 Score: {r2}")
```

## Model Performance

| Metric        | Score |
| :------------ | ----: |
| **R² Score**  |  0.86 |
| **Mean Absolute Error (MAE)** | 67,620 |
| **Root Mean Squared Error (RMSE)**    |  90,705 |

The model achieved high accuracy and effectively captured key patterns in property pricing.

## Feature Importance

Top factors influencing house prices:

1. Size in SqFt
2. Bedrooms 
3. Bathrooms
4. Property Type
5. Year Built

These insights help real estate agents prioritize what drives value and pricing in their market.

## Insights & Recommendations

1. Invest in Key Zip Codes: Focus on high-value neighborhoods with growth potential.
2. Property Improvements: Renovation and upgrades in condition or grade significantly boost property value.
3. Customer Advisory Tool: Deploy a predictive pricing tool to help clients estimate property worth.
4. Pricing Strategy: Use predictive insights to set competitive yet profitable listing prices.

## Tech Stack

* Language: Python
* Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
* Model: Random Forest Regressor
* Environment: Jupyter Notebook

## Next Steps

* Implement XGBoost for improved prediction accuracy.
* Develop a Streamlit web app for real-time house price predictions.
* Integrate geospatial data (latitude/longitude) to enhance location-based modeling.

## Author

**Holiness Segun-Olufemi**
Public Policy Professional | Data Scientist | Researcher
