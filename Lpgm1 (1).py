# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv("housing_prices_SLR.csv")  # assuming ',' is default delimiter

# Preview data
print(df.head())

# Access AREA and PRICE columns
print(df['AREA'])
print(df['PRICE'])

# Basic scatter plot
plt.scatter(df['AREA'], df['PRICE'], c='blue')
plt.title("Housing Prices vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Scatter plot with random colors
plt.scatter(df['AREA'], df['PRICE'], c=np.random.random(df.shape[0]))
plt.title("Random Colored Scatter Plot")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Scatter with random color and small size
col = np.random.random(df.shape[0])
plt.scatter(df['AREA'], df['PRICE'], c=col, s=4)
plt.title("Random Color + Small Size")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Step 3: Feature matrix and target vector
x = df[['AREA']].values  # feature matrix (2D)
y = df['PRICE'].values   # target vector (1D)

# Preview
print("First 5 x values:\n", x[:5])
print("First 5 y values:\n", y[:5])

# Step 4: Split the data into 80-20
from sklearn.model_selection import train_test_split
print("Any NaNs?\n", df.isna().sum())
df = df.dropna()
x = df[['AREA']].values
y = df['PRICE'].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# Step 5: Fit the line (Train the SLR model)
from sklearn.linear_model import LinearRegression

# Model with intercept
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

print("Intercept with intercept=True:", lr_model.intercept_)
print("Coefficient with intercept=True:", lr_model.coef_)

# Model without intercept
lr_model = LinearRegression(fit_intercept=False)
lr_model.fit(x_train, y_train)

print("Intercept with intercept=False:", lr_model.intercept_)
print("Coefficient with intercept=False:", lr_model.coef_)

# Step 6: Predict using the model
predicted_values = lr_model.predict(np.array([[2000], [2500]]))
print("Predicted prices for area 2000 and 2500:", predicted_values)

# Step 7: Calculating R² score
from sklearn.metrics import r2_score

# R² for training set
r2_train = r2_score(y_train, lr_model.predict(x_train))
print("R² score (train):", r2_train)

# R² for test set
r2_test = r2_score(y_test, lr_model.predict(x_test))
print("R² score (test):", r2_test)

# Alternative: using model's built-in score method
print("R² score using model.score:", lr_model.score(x_test, y_test))

# Step 8: Visualizing the model
plt.scatter(x_train[:, 0], y_train, c='red', label='Train Data')
plt.scatter(x_test[:, 0], y_test, c='blue', label='Test Data')

# Plot regression line
plt.plot(x_train[:, 0], lr_model.predict(x_train), c='yellow', label='Model Prediction')

plt.title("Linear Regression Fit")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()
