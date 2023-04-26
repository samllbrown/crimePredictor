import folium
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import webbrowser

tf.random.set_seed(None)

def clean_data(data):
    data = data[data['Crime ID'] != '']
    data = data[data['Latitude'].notnull()]
    data = data[data['Longitude'].notnull()]
    data = data[data['Longitude'] != 0]
    data = data[data['Latitude'] != 0]
    data = data.dropna(subset=['Longitude', 'Latitude'])
    return data[['Latitude', 'Longitude']]

def clean_data_for_rf(data):
    data = data[data['Crime ID'] != '']
    data = data[data['Latitude'].notnull()]
    data = data[data['Longitude'].notnull()]
    data = data[data['Longitude'] != 0]
    data = data[data['Latitude'] != 0]
    data = data.dropna(subset=['Longitude', 'Latitude', 'Crime type'])
    return data[['Latitude', 'Longitude', 'Crime type']]

crime_type_colour_map = {
    'Anti-social behaviour': 'blue',
    'Bicycle theft': 'green',
    'Burglary': 'purple',
    'Criminal damage and arson': 'red',
    'Drugs': 'darkgreen',
    'Public order': 'orange',
    'Robbery': 'pink',
    'Shoplifting': 'cadetblue',
    'Theft from the person': 'darkpurple',
    'Vehicle crime': 'darkred',
    'Violence and sexual offences': 'black',
    'Other crime': 'gray',
    'Other theft': 'lightred',
    'Possession of weapons': 'beige'
}


def generate_crime_map(center_latitude, center_longitude, crime_locations, crime_types, crime_type_colour_map):
    crime_map = folium.Map(location=[center_latitude, center_longitude], zoom_start=12)

    for (lat, lon), crime_type in zip(crime_locations, crime_types):
        color = crime_type_colour_map[crime_type]
        folium.Marker(location=[lat, lon], popup=f"Predicted Crime Type: {crime_type}",
                      icon=folium.Icon(color=color)).add_to(crime_map)

    return crime_map

data = pd.read_csv('training.csv')
data = clean_data(data)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X = data_scaled
y = data_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

input_data = pd.read_csv('training.csv')
input_data = clean_data(input_data)
input_data_scaled = scaler.transform(input_data)

predicted_crime_locations = model.predict(input_data_scaled)
predicted_crime_locations = scaler.inverse_transform(predicted_crime_locations)

data_rf = pd.read_csv('training.csv')
data_rf = clean_data_for_rf(data_rf)

scaler_rf = MinMaxScaler()
data_rf[['Latitude', 'Longitude']] = scaler_rf.fit_transform(data_rf[['Latitude', 'Longitude']])

label_encoder = LabelEncoder()
data_rf['Crime type'] = label_encoder.fit_transform(data_rf['Crime type'])

X_rf = data_rf[['Latitude', 'Longitude']]
y_rf = data_rf['Crime type']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

smote = SMOTE(random_state=42)
X_train_resampled_rf, y_train_resampled_rf = smote.fit_resample(X_train_rf, y_train_rf)

clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf.fit(X_train_resampled_rf, y_train_resampled_rf)

predicted_crime_locations_scaled = scaler_rf.transform(predicted_crime_locations)
predicted_crime_types_encoded = clf.predict(predicted_crime_locations_scaled)
predicted_crime_types = label_encoder.inverse_transform(predicted_crime_types_encoded)

south_latitude = 51.59
north_latitude = 51.67
west_longitude = -4.05
east_longitude = -3.85

filtered_crime_locations = []
filtered_crime_types = []

for i, coord in enumerate(predicted_crime_locations):
    if south_latitude <= coord[0] <= north_latitude and west_longitude <= coord[1] <= east_longitude:
        filtered_crime_locations.append(coord)
        filtered_crime_types.append(predicted_crime_types[i])

center_latitude = np.mean([coord[0] for coord in filtered_crime_locations])
center_longitude = np.mean([coord[1] for coord in filtered_crime_locations])

crime_map = generate_crime_map(center_latitude, center_longitude, filtered_crime_locations, filtered_crime_types, crime_type_colour_map)
crime_map.save("swansea_predicted_crime_map.html")
webbrowser.open("swansea_predicted_crime_map.html")

