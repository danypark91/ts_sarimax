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

# Decomposing the Positive cases
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

from pylab import rcParams
rcParams['figure.figsize'] = 18, 10

df_inf_pos = df_inf["Number_Positive"]

decomp = seasonal_decompose(df_inf_pos,period=7)
decomp.plot();