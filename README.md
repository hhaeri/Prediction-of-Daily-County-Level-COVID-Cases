# Prediction of Daily County Level COVID Cases in School Age Population using Machine Learning
## Hanieh Haeri
### 1. Project Motivation

Using the COVID-19 case-count data of all the counties in California, I built a classical machine learning model to predict the daily county-level case-count in school age population in California. This project uses timeseries forecasting through Auto Regressive Integrated Moving Average (ARIMA) model to provide prediction ten days in to future. The result of this model have the potential to provide insights on better preparedness for COVID-19 school resource allocation as State and School officials can use the models’ predicted case-counts to allocate the COVID-19 preventation resources more efficiently by sending them where they are most needed. The model results also can be used to determine the timing of implementation and relief of social distancing policies and safety regulations at individual counties and school districts.
### 2. Data Extraction/Munging and Feature Engineering

#### 2.1 Data Selection and Extraction

#### COVID-19 Case Surveillance Restricted Access Detailed Data
The COVID-19 case surveillance system database includes patient-level data reported to U.S. states and autonomous reporting entities, including New York City and the District of Columbia (D.C.), as well as U.S. territories and states. These data include demographic characteristics, exposure history, disease severity indicators and outcomes, clinical data, laboratory diagnostic test results, and comorbidities. The restricted access data set includes 32 features, however only limited number of the features were essential to this project. In this project I only use the following attributes of the dataset: 

- **The earlier of the Clinical Date**:Cdc_case_earliest_dt uses the best available date from both cdc_received_dt and cdc_clinical_obs_dt and is an option to end-users who need a date variable with optimized completeness. The logic of cdc_case_earliest_dt is to use the non-null date of one variable when the other is null and to use the earliest valid date when both dates are available.
- **Case status**
- **Age group** (0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+ years)
- **State of residence**
- **County of residence**

This is a **private** dataset with **restricted** access. CDC granted me the access to the GitHub repository after a request was sent to CDC. 

This data set updates the **first Tuesday of every month**. Since February 2021, the data was available in both CSV and **Parquet** format, both Parquet and CSV files are identical. I experienced faster data extraction using the Parquet format,hence the Parquet format is used in my project. Note, that at the earlier folders I notice that the Parquet files are in zipped format, so in future we need to make sure unzip them if they come in zipped format.

For the herein project, the following filters were performed: 
* The data was filtered to only include the confirmed cases (Selected Case Status = Laboratory-confirmed case)
* The data was filtered for the State of California (Selected State of residence = California)
* The cases outside the school-age populations were filtered (Selected Age group = 0-9 and 10-19) 

After the desired data is collected from the original dataset, the cases were aggregated with daily frequency for each county and the daily count of the confirmed cases in school age population at individual counties in Califiornia are calculated. This data along with the normalized case count (case count per 100K) are used later for forcasting county-level case count of California school age population using timeseries modeling. 

#### Cencus Bureau Data

Annual Estimates of the Resident Population for Counties is availabe through the Cencus Bureau Dataset. The most recent estimate of the county-level detailed population data was fetched from the Cencus Bureau website and used in this project.  

#### 2.2 Data Munging and Feature Engineering

Timeseries modeling requires the data to have consistant time intervals, such that if the model is given daily timeseries it can not have missing periods in it. In fact ARIMA has no understanding of the actual dates associated with each data, but instead it only given a data series. Clearly speaking, if data is missing for Tuesday and the model is given daily data, the Wednesday data is read as the missing data of Tuesday and so on. So, in order to avoid this problem we need to provide the model with a complete dataset, such that there is no hole or missing period in the dataset. There are many differernt options how one can overcome this problem. In this project I am assigning zero values to all times with no case reported. The logic behind this was that, I assume if there is no case reported to CDC it means that there were no actual confirmed case happening that specific day. This seems like a good estimate for counties with little number of days with no case but gets a bit iffy with the ones with many days with no reported case, is the data missing or not reported at all?  

Let's take a look at the missing case count data for the California counties. To this aim, I use Missingno library in python that is a very helpful tool for visualizing incompleteness in a dataset, it works on top of Matplotlib and Seaborn and is effortless to use. The following plot visualize the missing school age case counts for each California county. Each bar represent a county as labeled at the top and the fullness of the bar depicts how the data is populated (white: sparse black: complete).

![image](https://user-images.githubusercontent.com/91407046/161200807-03ea54bc-6fa1-4c80-a303-997b6995514e.png)

At a glance we can see which counties appear to be completely populated as opposed to the ones rarely populated (mostly black versus mostly white bars). With a close look at the missing values matrix we can identify the counties with sparse case count reported to CDC. The following table lists the counties with more than 40% missing case counts.

![image](https://user-images.githubusercontent.com/91407046/161211978-653420b8-f03c-411a-8257-7cf305023945.png)

It is very important to understand that the model is not going to be reliable for these counties as the data is sparse (many zero values) and the ARIMA model is not going to be able to predict the timeseries patern and will perform a lousy fit, not because of any problem with the model but the lack of data.

### 3. Timeseries forecasting through ARIMA

Using the COVID-19 case-count data of all the counties in California, I built a classical machine learning model to predict the spread of COVID-19 throughout the California counties. This project uses timeseries forecasting through Auto Regressive Integrated Moving Average (ARIMA) model to predict the spread of COVID-19 throughout the California counties ten days in advance. The result of this small-scale model have the potential to provide insights on better preparedness for COVID-19 as State and School officials can use the models’ predicted case-counts to allocate the COVID-19 preventation resources more efficiently by sending them where they are most needed. The model results also can be used to determine the timing of implementation and relief of social distancing policies and safety regulations at individual counties.

#### 3.1 Train_Test Split

The prediction time frame for the ARIMA model used 80% of the data as its training set while the remaining 20% was used as the prediction test set. This results in an approximately 5 month prediction period for the test data. More specifically, 618 days from January 01, 2020 to September 11, 2021 were used as the training period, while the remaining 155 days from September 12, 2021 to February 13, 2022 were used as the prediction test period. After the model is fitted using the train dataset, it is going to be used to make forecast ten days in to future that is until February 23, 2022.

It should be noted that due to the nature of COVID, unless the train period contains the same shock peaks of the different COVID-19 variants as the test period, it is not possible to provide a decent prediction of the test period based on the fitted model of the train period. That is the case with our model, such that the most recent COVID19 variant and its associated high peak occurs completely in the test period (test period starts at September 12, 2022, Omicron first confirmed case in California: November 09, 2022). Thus, if the model is trained using a traditional train-test split method, the trained model is not trained to predict the Omicron shock. For this reason, my ARIMA model is designed to predict one time step at a time in the test period. After the prediction is performed for that time step, the true value of the test at that step is moved to the train set. This continues until the end of the test period.   

#### 3.2. Hyperparameters

Hyperparameter tuning is an essential aspect of the training process of any machine learning model. A hyperparameter is any parameter typically in the form of a variable whose value is used in machine learning to influence the learning process. 

An ARIMA model has three hyperparameters to tune: 
* **p**: autoregressive lag order: number of previous values to be used
* **d**: degree of differencing: number of non-seasonal differences needed for stationarity
* **q**: moving average window size: number of data points used to compute a weighted average

![image](https://user-images.githubusercontent.com/91407046/161170708-45ac26d3-4d9e-44ee-a852-e0f1ae6b36fe.png)

The final hyperparameters selected for my model is a combination of (5,1,0),that is the lag order 5, the degree of differencing 1, and the moving average window size 0. these hyperparameters were fine tuned through trial-and-error for all counties. The combination with the best results was selected and these hyperparameters were used for the models. 

# 4. Model Results Visualization

For this project I created a heroku app that provides the user with an interactive Alter chart that plots the actual case count versus the model predicted count for the user's desired county picked from the drop down menu at the bottom of the chart. Here you can explore the app:

https://ca-covid-schoolage-hhaeri.herokuapp.com/

The user can look at the true daily case count versus the predicted value at anytime starting from the very first covid case that apeared in California till now or zoom in to specific time periods to get a closer look. The following chart shows an snapshot of the interactive plot explained here for the Contra Costa county.

![image](https://user-images.githubusercontent.com/91407046/161214467-1ae34b18-709e-48f3-88cc-b28a2e83ed9a.png)

The user also can zoom in to the more recent times and see the forecasted case counts for the desired county 10 days in to the future. 

![image](https://user-images.githubusercontent.com/91407046/161214807-5d987810-d3f7-4d56-a6a5-f8308cda1554.png)

Another visual that is provided to the user through my Heroku app is the map of Califonia counties color coded with their average 10-days case count forecast for the school age population. The map is labled to show the counties with top 10 average forecast value. This visual can provide quick insight to State and school officials on better allocation of COVID-19 prevetation school resources. 

![image](https://user-images.githubusercontent.com/91407046/161219208-07182207-f22b-47c1-bbb1-aeff751a5c40.png)

# 5. The app
This repository also contains files that I used to create the app  for this project:

https://ca-covid-schoolage-hhaeri.herokuapp.com/

The app is created in streamlit and deployed to heroku. It plots true and predicted values of daily confirmed COVID cases for the user_selected county in state of California for the school age population as explained in **Section 4**. The prediction was performed through timeseries forecasting using ARIMA as explained in **Section 3**.
