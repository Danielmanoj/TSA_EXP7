# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 

### Developed by : MANOJ G
### Register no :212222240060

### AIM:
To Implementat an Auto Regressive Model using Astrobiological dataset.
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# Load the CSV file
df = pd.read_csv('/content/astrobiological_activity_monitoring.csv')

# Display the first few rows of the dataset to see the data in row and column format
print("Initial Data Preview:")
print(df.head())

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Perform Augmented Dickey-Fuller test for stationarity on Soil_Microbial_Activity
result = adfuller(df['Soil_Microbial_Activity'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Plot ACF and PACF
plt.figure(figsize=(10,5))
plt.subplot(121)
plot_acf(df['Soil_Microbial_Activity'], lags=13, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Soil_Microbial_Activity'], lags=13, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df['Soil_Microbial_Activity'][:train_size], df['Soil_Microbial_Activity'][train_size:]

# Fit AutoRegressive model with 13 lags
model = AutoReg(train, lags=13)
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the test data and the predictions
plt.plot(test.index, test, label='Actual', marker='o')
plt.plot(test.index, predictions, label='Predicted', marker='x')
plt.title('Soil Microbial Activity - Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Soil Microbial Activity')
plt.legend()
plt.show()

```

### OUTPUT:


PACF - ACF :

![image](https://github.com/user-attachments/assets/67f7a813-ead7-4160-8d3d-d5878ce63f25)



PREDICTION :

![image](https://github.com/user-attachments/assets/86cbf2bc-a375-4b07-9b60-148d6f74e452)


### RESULT:
Thus the python code successfully implemented the auto regression function.
