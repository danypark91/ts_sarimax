# ts_sarimax
Time series analysis of Flu dataset using Pandas using Seasonal ARIMA model.

### Tech/Framework used
- Jupyter Notebook (.ipynb)
- PyCharm (.py)

### Dataset
The [dataset](https://data.world/chhs/fc544658-35c5-4be0-af20-fc703bc57c13) contains for California influenza surveillance data. The dataset for this article is from [data.world](https://data.world/chhs/fc544658-35c5-4be0-af20-fc703bc57c13/workspace/file?filename=clinical-sentinel-laboratory-influenza-and-other-respiratory-virus-surveillance.csv). The particular dataframe has 8 attributes, summarizing weekly incidents of the influenza infections.

### Description
This project is to apply Seasonal ARIMA to the dataset. It begins with the importation of the dataset from the direct url from the data.world and checks if it requires data cleansing. The cleansed weekly data gets resampled into the monthly summary and divided into **train** and **test sets**. The best-fit model gets derived by using **train_df**. The model undergoes statistical tests to determine scientific accuracy and applied to the test_df to check the predictability of the models. Finally, the model is used to predict next 24 incidents (months) with it's 95% confidence intervals.

### Reference
- [Forecasting: Principles and Practice](https://otexts.com/fpp2/seasonal-arima.html)
- [Penn State - STAT510](https://online.stat.psu.edu/stat510/lesson/4/4.1)
- https://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
