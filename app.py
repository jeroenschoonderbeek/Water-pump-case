import streamlit as st
import numpy as np
import datetime
import pandas as pd

import plotly.graph_objects as go

### Define Titles ###
st.title("Water pump dashboard")
st.write("### Big Data Replublic Data Science Case")

st.sidebar.markdown("## Controls")
st.sidebar.markdown("You can adjust the control inputs to adjust the prediction")

### Define input controls ###
st.write("##### Input Controls ")

st.write("### Numerical input Controls ")
amount_tsh = st.sidebar.number_input('Total static head (amount water available to waterpoint)', value=350)
st.write(f"Total static head (amount water available to waterpoint)={amount_tsh}")

gps_height = st.sidebar.number_input('Altitude of the well', value=668)
st.write(f"Altitude of the well)={gps_height}")

longitude  = st.sidebar.number_input('Longitude GPS coordinate', value=34.08)
st.write(f"Longitude GPS coordinate)={longitude}")

latitude  = st.sidebar.number_input('Latitude GPS coordinate', value=-5.7)
st.write(f"Latitude GPS coordinate)={latitude}")

population  = st.sidebar.number_input('Population around the well', value=180)
st.write(f"Population around the well={population}")

construction_year  = st.sidebar.number_input('Year the waterpoint was constructed', value=1300)
st.write(f"Year the waterpoint was constructed={construction_year}")

st.write("### Categorical input Controls ")
extraction_type_class = st.sidebar.radio(
     "The kind of extraction the waterpoint uses",
     ['gravity', 'handpump', 'other', 'submersible', 'motorpump', 'rope pump'])

st.write(f"The kind of extraction the waterpoint uses={extraction_type_class}")

management_group = st.sidebar.radio(
     "How the waterpoint is managed",
     ['user-group', 'commercial', 'parastatal', 'other', 'unknown'])

st.write(f"How the waterpoint is managed={management_group}")

payment = st.sidebar.radio(
     "What the water costs",
     ['never pay', 'pay per bucket', 'pay monthly', 'unknown', 'pay when scheme fails', 'pay annually'])

st.write(f"What the water costs={payment}")

water_quality = st.sidebar.radio(
     "The quality of the water",
     ['soft', 'salty', 'unknown', 'milky', 'coloured', 'salty abandoned'])

st.write(f"The quality of the water={water_quality}")

quantity = st.sidebar.radio(
     "The quantity of water",
     ['enough', 'insufficient', 'dry', 'seasonal', 'unknown'])

st.write(f"The quantity of water={quantity}")

source = st.sidebar.radio(
     "The source of the water",
     ['spring', 'shallow well', 'machine dbh', 'river', 'rainwater harvesting', 'hand dtw'])

st.write(f"The source of the water={source}")


### Make predictions ###
feature_input = {
    'amount_tsh':amount_tsh,
    'gps_height':gps_height ,
    'longitude':longitude,
    'latitude':latitude,
    'population':population,
    'construction_year':construction_year,
    'extraction_type_class':extraction_type_class,
    'management_group':management_group,
    'payment':payment,
    'water_quality':water_quality,
    'quantity':quantity,
    'source':source,
}

# Transform input to dataframe
prediction_data = pd.DataFrame(feature_input, index=[0])

# Encode cat features to dummies
categorical_features = [
    'extraction_type_class',
    'management_group',
    'payment',
    'water_quality',
    'quantity',
    'source',
]

for cat_feature in categorical_features:
    
    # Hoe ontwikkelen de verschillende failures zich over tijd?
    cat_dummies = pd.get_dummies(prediction_data[cat_feature])
    
    cat_dummies.columns = cat_feature + '_' + cat_dummies.columns
            
    prediction_data = prediction_data.join(cat_dummies)

# Define features
features = [
        'amount_tsh',
        'gps_height',
        'longitude',
        'latitude',
        'population',
        'construction_year',
        'extraction_type_class_gravity',
        'extraction_type_class_handpump',
        'extraction_type_class_motorpump',
        'extraction_type_class_other',
        'extraction_type_class_rope_pump',
        'extraction_type_class_submersible',
        'extraction_type_class_wind-powered',
        'management_group_commercial',
        'management_group_other',
        'management_group_parastatal',
        'management_group_unknown',
        'management_group_user-group',
        'payment_never_pay',
        'payment_other',
        'payment_pay_annually',
        'payment_pay_monthly',
        'payment_pay_per_bucket',
        'payment_pay_when_scheme_fails',
        'payment_unknown',
        'water_quality_coloured',
        'water_quality_fluoride',
        'water_quality_fluoride_abandoned',
        'water_quality_milky',
        'water_quality_salty',
        'water_quality_salty_abandoned',
        'water_quality_soft',
        'water_quality_unknown',
        'quantity_dry',
        'quantity_enough',
        'quantity_insufficient',
        'quantity_seasonal',
        'quantity_unknown',
        'source_dam',
        'source_hand_dtw',
        'source_lake',
        'source_machine_dbh',
        'source_other',
        'source_rainwater_harvesting',
        'source_river',
        'source_shallow_well',
        'source_spring',
        'source_unknown'
            ]

# If feature value is missing set to 0
for f in features:
    if f not in prediction_data.columns:
        prediction_data[f] = 0

### Predict probabilities per class
import json
import requests
scoring_uri = ("http://07492e3c-071b-4fbf-b398-28c4255781cc.northeurope.azurecontainer.io/score")
raw_data = prediction_data[features].values
raw_data_list = raw_data.tolist()
json_format = {"data": raw_data_list}
input_data = json.dumps(json_format)
headers = {"Content-Type": "application/json"}
resp = requests.post(scoring_uri, input_data, headers=headers)
import ast
import re
prediction = ast.literal_eval(resp.text)[0]

# Write predictions to dashboard
st.write("### Status group predictions for given control inputs")
function_probability = str(round((prediction[0] * 100), 1)) + " %"
repair_probability = str(round((prediction[1] * 100), 1)) + " %"
non_function_probability = str(round((prediction[2] * 100), 1)) + " %"

st.write(f"Probability water pump is functional={function_probability}")
st.write(f"Probability water pump is functional but needs repair={repair_probability}")
st.write(f"Probability water pump is non functional={non_function_probability}")

### XAI ###
url_xai = "http://46482af5-d55f-4318-9c38-3997d99ec076.northeurope.azurecontainer.io/score"

raw_data = prediction_data[features].values
raw_data_list = raw_data.tolist()
json_format = {"data": raw_data_list}
input_data = json.dumps(json_format)
headers = {"Content-Type": "application/json"}
resp = requests.post(url_xai, input_data, headers=headers)
result_xai_df = pd.DataFrame(json.loads(resp.text))

result_xai_df.reset_index(drop=True, inplace=True)

new_data = []
# Show only 15 most important features for predictions
max_contrib = 15

# Transform XAI predictions
for index, row in result_xai_df.iterrows():
    for i in range(1, max_contrib):
        value = prediction_data[row['feature_' + str(i)].lower()].iloc[index]
        try:
            value = round(float(value), 2)
        except:
            pass

        feature = row['feature_' + str(i)]

        new_data.append({
            'class_prediction': row['ypred'],
            'feature': row['feature_' + str(i)], 
            'contribution': row['contribution_' + str(i)], 
            'value': value,
        })
xai_df = pd.DataFrame(new_data, columns=['class_prediction', 'feature', 'contribution', 'value'])

# Plot XAI results in bar chart
import plotly.graph_objects as go

xai_df = xai_df.sort_values(by=['contribution'])
fig = go.Figure()
fig.add_trace(go.Bar(
    y=xai_df['feature'].values,
    x=xai_df['contribution'].values,
    name='Feature Contribution',
    orientation='h',
    marker=dict(
        color='rgba(246, 78, 139, 0.6)',
        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
    )
))

fig.update_layout(
    barmode='stack',
    title_text='Feature contribution to prediction based on class with highest probability', # title of plot
    xaxis_title_text='Feature contribution to base value 0.85', # xaxis label
    yaxis_title_text='Feature', # yaxis label)
)
st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

### General feature importance XAI Model ###
fig = go.Figure()
fig.add_trace(go.Bar(
    y=['quantity_enough', 'population', 'extraction_type_class_gravity','payment_never_pay', 'amount_tsh', 'construction_year', 'latitude','extraction_type_class_other', 'longitude', 'quantity_dry'],
    x=[2.98, 3.06, 3.56, 3.77, 5.01, 5.74, 7.32, 7.42, 7.57,24.9 ],
    name='Feature Contribution',
    orientation='h',
    marker=dict(
        color='rgba(246, 78, 139, 0.6)',
        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
    )
))

fig.update_layout(
    barmode='stack',
    title_text='Top 10 feature importance of ML model in general', # title of plot
    xaxis_title_text='Feature importance in percentage (%)', # xaxis label
    yaxis_title_text='Feature', # yaxis label)
)
st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
