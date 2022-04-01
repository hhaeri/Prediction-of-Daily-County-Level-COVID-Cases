# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:32:36 2022

@author: hhaeri
"""
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
import streamlit as st
import altair as alt
import geopandas


#############################################################################
st.set_page_config(
      page_title="hhaeri_capstone_app",
      page_icon=":)",
      layout="wide",
      initial_sidebar_state="expanded",
      menu_items={
          'Get Help': 'https://github.com/hhaeri/TDI_Capstone',
          'About': "# Welcome to my covid case presiction app!"
      }
  )

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
.medium-font {
    font-size:20px !important;
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

#############################################################################
#Configuring the title and sidebar for the app created by streamlit

#st.title("The Data Incubator Capstone Project")
st.markdown('<p class="big-font">The Data Incubator Capstone Project </p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">Hanieh Haeri </p>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="medium-font">The Data Incubator Capstone Project</p> Hanieh Haeri </p>', unsafe_allow_html=True)


st.sidebar.markdown('<p class="small-font">This is a simple app created by streamlit. It uses Python Requests libarary\
                    along with Census Bureau and CDC Github Data and plots confirmed COVID cases for a selected county \
                    in state of California versus my ARIMA model prediction. ARIMA model in a machine learning setting is used\
                    to predict future values of COVID cases in individual counties in California. </p>                  \
                    <p>Scikit-learn has limited capabilities for dealing with time series data. In order to utilize the more\
                    rich ARIMA family of models, I use pmdarima package in Python.</p> <p>For more information, visit my \
                    Git Hub at https://github.com/hhaeri/TDI_Capstone</p> \
                    Comments? <p>Email: hhaeri0911@gmail.com</p>'
                    , unsafe_allow_html=True)
###########################################################################


# Get the pickled model prediction
prediction_df = pd.read_pickle("./df_CA_prediction.pkl")

#Prepare the dataframe for the plotting
df_4_plot = pd.melt(prediction_df.drop('type',axis = 1).drop('model',axis = 1),id_vars=['Date','County'], \
                    value_vars=None, var_name=None, value_name='Daily Case Counts', col_level=None, ignore_index=True)

    
alt.data_transformers.disable_max_rows()
# Define a new selection object.
brush = alt.selection(type='interval', encodings=['x'], clear=False)


# Link the domain selected by brush to the X range of the chart.
base = alt.Chart(df_4_plot, width=400, height=400).mark_point(filled=True).encode(
    alt.X('Date:T', axis = alt.Axis(title = 'Date'.upper(),labelAngle=90, format = ("%d %b %Y"), labelFontSize=15,titleFontSize = 15), scale=alt.Scale(domain=brush)),
    alt.Y('Daily Case Counts', axis = alt.Axis(title = 'Daily COVID Cases',titleFontSize = 15,labelFontSize=15)),
    alt.Color('variable'),
    alt.Shape('variable', scale=alt.Scale(range=['cross']), legend=None)
)

columns = df_4_plot['County'].unique()

# A dropdown filter
column_dropdown = alt.binding_select(options=columns)
column_select = alt.selection_single(
    fields=['County'],
    on='doubleclick',
    clear=False, 
    bind=column_dropdown, 
    name='y_value',
    init={'County': 'CONTRA COSTA'}
)


# Specify the top chart as a modification of the base chart
filter_columns = base.add_selection(
    column_select
).transform_filter(
    column_select
).properties(
    height=340,
    width=1000
)


# Specify the lower chart as a modification of the base chart
lower = alt.Chart(df_4_plot).mark_line().encode(
    alt.X('Date:T', axis = alt.Axis(title = 'Date'.upper(),labelAngle=90, format = ("%b %Y"))),
    alt.Y('Daily Case Counts', title=' '),
    alt.Color('variable')
).add_selection(
    column_select
).transform_filter(
    column_select
).properties(
    height=60,
    width=1000
).add_selection(brush)


chart_data = filter_columns & lower

st.altair_chart(chart_data, use_container_width=True)

#############################################################
#Now plotting the map
#PAss the map to altair
#alt.renderers.enable('notebook')

# Get the population data that will be used later with the map
response2 = requests.get('https://api.census.gov/data/2019/acs/acs5/profile?get=NAME,DP05_0019E&for=county:*&in=state:06')
under18_population_data = response2.json()
df_population_under18 = pd.DataFrame(under18_population_data[1:],columns=under18_population_data[0])\
                        .drop(['state','county'],axis = 1).rename(columns={'NAME':'res_county'})\
                        .rename(columns = {'DP05_0019E':'population_under18'})
df_population_under18['res_county'] = df_population_under18['res_county'].str.upper().str.replace(' COUNTY, CALIFORNIA', '')                        
                        
df_forecast = df_4_plot.loc[(df_4_plot['variable']=='prediction')&(df_4_plot['Date']==df_4_plot['Date'].max())]\
                        [['County','Daily Case Counts']].\
                        merge(df_population_under18.rename(columns ={'res_county':'County'}),on='County')
df_forecast = df_forecast.apply(pd.to_numeric, errors='ignore')
df_forecast['Daily_Case_per100k_under18']=(df_forecast['Daily Case Counts']*100000./(df_forecast['population_under18']*1.)).\
    apply(np.ceil)
df_forecast['Daily Case Counts'] = df_forecast['Daily Case Counts'].apply(np.ceil)
df_forecast['Daily Case Counts _Normalized'] = (df_forecast['Daily Case Counts']-(df_forecast['Daily Case Counts'].mean()))\
                                                /(df_forecast['Daily Case Counts'].std())
df_forecast['rank']=df_forecast['Daily Case Counts _Normalized'].rank(method = 'max',ascending = False)

#Create a column that contains the county names for the ones with top 10 case count prediction, 
# this is going to be used for the map
df_forecast['County_High_Ranks'] = df_forecast[df_forecast['rank']<11]['County']
df_forecast.fillna('',inplace = True)  

#Now get the base map of the US counties
counties = geopandas.read_file('../cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
CA_counties=counties[counties['STATEFP']=='06'].rename(columns = {'NAME':'County'})
CA_counties['County'] = CA_counties['County'].str.upper()
#add the prediction to the basemap
CA_counties = CA_counties.merge(df_forecast,on='County')
CA_counties['County']=CA_counties.County.astype(str)


# map_data = alt.Chart(CA_counties).mark_geoshape().encode(color ='Daily Case Counts _Normalized', 
#                                               tooltip =['County','rank']).properties(width=500,height=300)
                                                                                                                             
                                                                                                                             
# st.altair_chart(map_data, use_container_width=True)
# st.map(CA_counties, use_container_width=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

gdf = geopandas.GeoDataFrame(CA_counties)
ax = CA_counties.plot(column = 'Daily Case Counts _Normalized',cmap='OrRd',vmax = 3.,figsize=(5, 5))
ax.axis('off')
CA_counties.apply(lambda x: ax.annotate(text=x['County_High_Ranks'], xy=x.geometry.centroid.coords[0], 
                                        ha='center', fontsize=3,bbox={'facecolor': 'none', 'alpha':0.8, 
                                                                      'pad': 1, 'edgecolor':'none'}),axis=1);
plt.title('Map of average 10-days forecast of Daily Case Counts \n in school age population in California counties', fontsize=4) 

st.pyplot()        
