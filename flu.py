# ----------------------------------------------------------------------------------------------------------- #
#                                                    Full Code                                                #
# ----------------------------------------------------------------------------------------------------------- #

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as stl
import datetime as dt
import seaborn as sns
import numpy as np

# Importing Data
flu = pd.read_csv("https://query.data.world/s/36nbd4wq3xmkqijc5a47ujr34h47og")
flu.head(5)

flu.describe()
flu.info()

# print unique value of every column of the dataset
for col in flu:
    print(flu[col].unique())

# converting weekending column into date
flu["weekending"]=pd.to_datetime(flu["weekending"]).dt.date
flu.sort_values(by="weekending").head(5)

piv_table = pd.pivot_table(flu, values="Number_Positive", index=["weekending"], columns=["Respiratory_Virus"])
piv_table

# Conditions to filter only total value
inf_istotal = flu["Respiratory_Virus"] == "Total_Influenza"
cov_istotal = flu["Respiratory_Virus"] == "Total_Coronavirus"

# Subset of flu for total_influenza
tot_infflu = flu[inf_istotal]
tot_infflu.head(5)

table1 = pd.pivot_table(tot_infflu, values="Number_Positive", index=["weekending","region"], fill_value=0)
table1

# Total sum by the date
inf_groupby=tot_infflu.groupby(["weekending"])["Number_Positive", "Specimens Tested"].sum()
df_inf = pd.DataFrame(inf_groupby)
df_inf["Positive_Rate"] = df_inf["Number_Positive"]/df_inf["Specimens_Tested"]
df_inf.head(5)

# Reset the index to DatetimeIndex for convenience
format ='%Y-%m-%d'
df_inf.index = pd.to_datetime(df_inf.index, format=format)
df_inf = df_inf.set_index(pd.DatetimeIndex(df_inf.index))

# Frequency of the Test and Positive Cases
df_inf.plot(figsize=(18,10),
            title="Weekly Flu case reported",
            xlabel="Date",
            ylabel="Frequency")
df_inf["Specimens_Tested"].plot()
plt.legend()

# Yearly Average plot
df_yearly = df_inf.resample("1Y").mean()

df_yearly["Number_Positive"].plot(figsize=(18,10),
                                 title="Yearly Average of Flu Cases",
                                 ylabel="Frequency")
df_yearly["Specimens_Tested"].plot()
plt.legend()

# Monthly Average plot
df_monthly = df_inf.resample("M").mean()

df_monthly["Number_Positive"].plot(figsize=(18,10),
                                 title="Monthly Average of Flu Cases",
                                 ylabel="Frequency")
df_monthly["Specimens_Tested"].plot()

plt.legend()

# Decomposing the Positive cases
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

from pylab import rcParams
rcParams['figure.figsize'] = 18, 10

df_monthly_pos = df_monthly["Number_Positive"]

decomp = seasonal_decompose(df_monthly_pos, model='additive')
decomp.plot();

# First Differencing: Stationarity
df_monthly_pos_diff = df_monthly_pos.diff()

fig, ax1 = plt.subplots(figsize=(18,8))
fig, ax2 = plt.subplots(figsize=(18,8))
fig, ax3 = plt.subplots(figsize=(18,8))

df_monthly_pos_diff.plot(ax=ax1, title="First Order Differencing");
plot_acf(df_monthly_pos_diff.dropna(), ax=ax2);
plot_pacf(df_monthly_pos_diff.dropna(), ax=ax3);

# Augmented Dicky-Fuller Test: Stationarity
from statsmodels.tsa.stattools import adfuller

def adfuller_result(y):
    # Dicky-Fuller test
    # If the Time Series is differenced prior,
    # drop the first cell of the series
    results = adfuller(y)

    # Parse the test and print the result
    print('ADF Statistics: %f' % results[0])
    print('p-value: %f' % results[1])
    print('Lags Used: %f' % results[2])
    print('Observations Used: %f' % results[3])
    print('Critical Values:')
    for key, value in results[4].items():
        print('\t%s: %.3f' % (key, value))

# Augmented Dicky-Fuller Test
adfuller_result(df_monthly_pos)

# Converting into the dataframe
df_inf_pos = df_monthly_pos.rename_axis('weekending').to_frame('Number_Positive')
df_inf_pos.head(5)

# Split test and train
from sklearn.model_selection import train_test_split
train_inf, test_inf = train_test_split(df_monthly_pos, test_size=0.2, shuffle=False)

# Visual representation of the split
plt.plot(train_inf)
plt.plot(test_inf)

# Best-fit of the model
import pmdarima as pm
model = pm.auto_arima(train_inf, d=1, D=1,
                      seasonal=True, m=7,
                      start_p=0, max_p=5,
                      start_q=0, max_q=5,
                      trace=True,
                      error_action='ignore',
                      supress_warning=True,
                      stepwise=True)

# Fit on train model
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_inf,
                order=(0,1,0),
                seasonal_order=(0,1,1,12),
               enforce_stationarity = False,
               enforce_invertibility = False)
result = model.fit()
result.summary()

# Plotting the dignostics of the train set
result.plot_diagnostics(figsize=(18,10))
plt.show()

# Fitting test-set with the model
inf_future = result.get_prediction(start=test_inf.index[0], dynamic=False)
inf_future_int = inf_future.conf_int()

# Plotting observed values and predictions
plt.figure(figsize=(18,10))

ax = df_monthly_pos.plot(label = "Number_Positive")
inf_future.predicted_mean.plot(ax=ax, label="Prediction", color = 'Red')
ax.fill_between(inf_future_int.index,
                inf_future_int.iloc[:, 1],
                color='grey', alpha=0.3, label = "Confidence Interval")

plt.ylabel("Number_Positive")
plt.legend()

# Positive Rate of the flu cases
df_inf['Positive_Rate'].plot(figsize=(18,10),
                               title="Weekly Flu case reported",
                               xlabel="Date",
                               ylabel="Rate")

# Positive rate of monthly resampled
df_monthly["Positive_Rate"].plot(figsize=(18,10),
                                 title="Monthly Positive Rate of Flu Cases",
                                 ylabel="Rate (%)")
plt.legend()

# Decomposing Positive Rate
from pylab import rcParams
rcParams['figure.figsize'] = 18, 10

df_monthly_pos2 = df_monthly["Positive_Rate"]

decomp = seasonal_decompose(df_monthly_pos2,model='additive')
decomp.plot();