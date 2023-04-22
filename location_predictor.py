import folium
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import webbrowser

# Set a random seed for TensorFlow to initialize the network weights differently each time
tf.random.set_seed(None)
def clean_data(data):
    data = data[data['Crime ID'] != '']
    data = data[data['Latitude'].notnull()]
    data = data[data['Longitude'].notnull()]
    data = data[data['Longitude'] != 0]
    data = data[data['Latitude'] != 0]
    data = data.dropna(subset=['Longitude', 'Latitude'])
    return data[['Latitude', 'Longitude']]

def generate_crime_map(center_latitude, center_longitude, crime_locations):
    crime_map = folium.Map(location=[center_latitude, center_longitude], zoom_start=12)

    for lat, lon in crime_locations:
        folium.Marker(location=[lat, lon], popup=f"Predicted Crime Location: {lat:.5f}, {lon:.5f}").add_to(crime_map)

    return crime_map

data = pd.read_csv('training.csv')
data = clean_data(data)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X = data_scaled
y = data_scaled

# Remove the random_state parameter from the train_test_split() function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

input_data = pd.read_csv('southwales12.csv')
input_data = clean_data(input_data)
input_data_scaled = scaler.transform(input_data)

predicted_crime_locations = model.predict(input_data_scaled[:8000])
predicted_crime_locations = scaler.inverse_transform(predicted_crime_locations)

south_latitude = 51.59
north_latitude = 51.67
west_longitude = -4.05
east_longitude = -3.85

filtered_crime_locations = [coord for coord in predicted_crime_locations
                            if south_latitude <= coord[0] <= north_latitude
                            and west_longitude <= coord[1] <= east_longitude]

center_latitude = np.mean([coord[0] for coord in filtered_crime_locations])
center_longitude = np.mean([coord[1] for coord in filtered_crime_locations])

crime_map = generate_crime_map(center_latitude, center_longitude, filtered_crime_locations)
crime_map.save("swansea_predicted_crime_map_large_bbox.html")
webbrowser.open("swansea_predicted_crime_map_large_bbox.html")
