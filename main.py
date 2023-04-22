from imblearn.over_sampling import SMOTE
import folium
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import webbrowser

def clean_data(data):
    data = data[data['Crime ID'] != '']
    data = data[data['Latitude'].notnull()]
    data = data[data['Longitude'].notnull()]
    data = data[data['Longitude'] != 0]
    data = data[data['Latitude'] != 0]
    data = data.dropna(subset=['Longitude', 'Latitude', 'Crime type'])
    return data[['Latitude', 'Longitude', 'Crime type']]

def generate_crime_map(center_latitude, center_longitude, crime_locations, crime_types, crime_type_colors):
    crime_map = folium.Map(location=[center_latitude, center_longitude], zoom_start=12)

    for (lat, lon), crime_type, color in zip(crime_locations, crime_types, crime_type_colors):
        folium.Marker(location=[lat, lon], popup=f"Predicted Crime Type: {crime_type}", icon=folium.Icon(color=color)).add_to(crime_map)

    return crime_map

data = pd.read_csv('training.csv')
data = clean_data(data)

scaler = MinMaxScaler()
data[['Latitude', 'Longitude']] = scaler.fit_transform(data[['Latitude', 'Longitude']])

label_encoder = LabelEncoder()
data['Crime type'] = label_encoder.fit_transform(data['Crime type'])

X = data[['Latitude', 'Longitude']]
y = data['Crime type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf.fit(X_train_resampled, y_train_resampled)


input_data = pd.read_csv('southwales12.csv')
input_data = clean_data(input_data)
input_data_scaled = scaler.transform(input_data[['Latitude', 'Longitude']])

predicted_crime_types_encoded = clf.predict(input_data_scaled[:8000])
predicted_crime_types = label_encoder.inverse_transform(predicted_crime_types_encoded)


crime_type_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

south_latitude = 51.59
north_latitude = 51.67
west_longitude = -4.05
east_longitude = -3.85

filtered_crime_locations = [coord for i, coord in enumerate(input_data[['Latitude', 'Longitude']].values)
                            if south_latitude <= coord[0] <= north_latitude
                            and west_longitude <= coord[1] <= east_longitude]

filtered_crime_types = [crime_type for i, crime_type in enumerate(predicted_crime_types)
                        if south_latitude <= input_data.iloc[i]['Latitude'] <= north_latitude
                        and west_longitude <= input_data.iloc[i]['Longitude'] <= east_longitude]

filtered_crime_type_colors = [crime_type_colors[label_encoder.transform([crime_type])[0]] for crime_type in filtered_crime_types]

center_latitude = np.mean([coord[0] for coord in filtered_crime_locations])
center_longitude = np.mean([coord[1] for coord in filtered_crime_locations])

crime_map = generate_crime_map(center_latitude, center_longitude, filtered_crime_locations, filtered_crime_types, filtered_crime_type_colors)
crime_map.save("swansea_predicted_crime_map.html")
webbrowser.open("swansea_predicted_crime_map.html")