# The Data Incubator Capstone Project
# Prediction of Daily County Level COVID Cases in School Age Population using Machine Learning
## Hanieh Haeri
### 1. Project Motivation

Using the COVID-19 case-count data of all the counties in California, I built a classical machine learning model to predict the daily county-level case-count in school age population in California. This project uses timeseries forecasting through Auto Regressive Integrated Moving Average (ARIMA) model to provide prediction ten days in advance. The result of this model have the potential to provide insights on better preparedness for COVID-19 school resource allocation as State and School officials can use the models’ predicted case-counts to allocate the COVID-19 preventation resources more efficiently by sending them where they are most needed. The model results also can be used to determine the timing of implementation and relief of social distancing policies and safety regulations at individual counties and school districts.
### 2. Data Extraction/Munging and Feature Engineering

#### 2.1 Data Selection and Extraction

#### COVID-19 Case Surveillance Restricted Access Detailed Data
The COVID-19 case surveillance system database includes patient-level data reported to U.S. states and autonomous reporting entities, including New York City and the District of Columbia (D.C.), as well as U.S. territories and states. These data include demographic characteristics, exposure history, disease severity indicators and outcomes, clinical data, laboratory diagnostic test results, and comorbidities. The restricted access data set includes the following variables: 

- **The earlier of the Clinical Date**:Cdc_case_earliest_dt uses the best available date from both cdc_received_dt and cdc_clinical_obs_dt and is an option to end-users who need a date variable with optimized completeness. The logic of cdc_case_earliest_dt is to use the non-null date of one variable when the other is null and to use the earliest valid date when both dates are available.

- Initial report date of case to CDC (Deprecated, use the earlier of the Clinical Date): This date was populated using the date at which a case record was first submitted to the database. If missing, then the report date entered on the case report form was used. If missing, then the date at which the case first appeared in the database was used.
- Date of first positive specimen collection
- Symptom onset date, if symptomatic
- **Case status**
- Sex
- **Age group** (0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+ years)
- Race and ethnicity (combined)
- **State of residence**
- **County of residence**
- County FIPS code
- Healthcare worker status
- Pneumonia present
- Acute respiratory distress syndrome (ARDS) present
- Abnormal chest x-ray (CXR) present
- Hospitalization status
- ICU admission status
- Mechanical ventilation (MV)/intubation status
- Death status
- Presence of each of the following symptoms: fever, subjective fever, chills, myalgia, rhinorrhea, sore throat, cough, shortness of breath, nausea/vomiting, headache, abdominal pain, diarrhea
- Presence of underlying comorbidity or disease

This is a **private** dataset with **restricted** access. CDC granted me the access to the GitHub repository after a request was sent to CDC. 

This data set updates the **first Tuesday of every month**. Since February 2021, the data was available in both CSV and **Parquet** format, both Parquet and CSV files are identical. I experienced faster data extraction using the Parquet format,hence the Parquet format is used in my project. Note, that at the earlier folders I notice that the Parquet files are in zipped format, so in future we need to make sure unzip them if they come in zipped format.

In this project I only use the following attributes of the dataset: 
* Date (The earlier of the Clinical Date), 
* Case Status, 
* Age group, 
* State and County of Residence

For the herein project, the following filters were performed: 
* The data was filtered to only include the confirmed cases (Selected Case Status = Laboratory-confirmed case)
* The data was filtered for the State of California (Selected State of residence = California)
* The cases outside the school-age populations were filtered (Selected Age group = 0-9 and 10-19) 

After the desired data is collected from the original dataset, the cases were aggregated with daily frequency for each county and the daily count of the confirmed cases in school age population at individual counties in Califiornia are calculated. This data along with the normalized case count (case count per 100K) are used later for forcasting county-level case count of California school age population using timeseries modeling. 

#### Cencus Bureau Data

Annual Estimates of the Resident Population for Counties is availabe through the Cencus Bureau Dataset. The most recent estimate of the county-level detailed population data was fetched from the Cencus Bureau website and used in this project.  

#### 2.2 Data Munging and Feature Engineering


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
