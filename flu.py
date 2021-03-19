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

# Positive Rate of the flu cases
df_inf['Positive_Rate'].plot(figsize=(18,10),
                               title="Weekly Flu case reported",
                               xlabel="Date",
                               ylabel="Rate")

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

df_inf_pos = df_inf["Number_Positive"]

decomp = seasonal_decompose(df_inf_pos, model='additive')
decomp.plot();

# Differencing: Stationarity
df_inf_pos_diff = df_inf_pos.diff()

fig, ax1 = plt.subplots(figsize=(18,8))
fig, ax2 = plt.subplots(figsize=(18,8))
fig, ax3 = plt.subplots(figsize=(18,8))

df_inf_pos_diff.plot(ax=ax1, title="First Order Differencing");
plot_acf(df_inf_pos_diff.dropna(), ax=ax2);
plot_pacf(df_inf_pos_diff.dropna(), ax=ax3);


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
adfuller_result(df_inf_pos)

