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
inf_groupby=tot_infflu.groupby(["weekending"])["Number_Positive"].sum()
df_inf = pd.DataFrame(inf_groupby)

