import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Read the dataset
data = pd.read_csv('training.csv')

data = data[data['Crime ID'] != '']
data = data[data['Latitude'].notnull()]
data = data[data['Longitude'].notnull()]
data = data[data['Crime type'].notnull()]
data = data[data['Longitude'] != 0]
data = data[data['Latitude'] != 0]
data = data.dropna(subset=['Longitude', 'Latitude'])
print("Number of rows in the dataset:", len(data))


locations = data[['Longitude', 'Latitude']]

# Train the K-means model
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(locations)

# Find the cluster centers (hotspots)
hotspots = kmeans.cluster_centers_

# Predict the next crime location

# Option 1: Choose the nearest hotspot to a given location
# given_location = np.array([[longitude, latitude]])
# nearest_hotspot = hotspots[np.argmin(cdist(given_location, hotspots), axis=1)]

# Option 2: Use a probability distribution based on the number of crimes in each hotspot
hotspot_counts = np.array([np.sum(kmeans.labels_ == i) for i in range(n_clusters)])
hotspot_probabilities = hotspot_counts / np.sum(hotspot_counts)

next_hotspot = hotspots[np.random.choice(n_clusters, p=hotspot_probabilities)]

import folium
import webbrowser

# Replace 'center_latitude' and 'center_longitude' with the coordinates of the center of your map
map = folium.Map(location=[51.624915, -3.940063], zoom_start=13)

# Add markers for each historical crime location
for index, row in locations.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], icon=folium.Icon(color='blue')).add_to(map)

# Add markers for the hotspots
for hotspot in hotspots:
    folium.Marker([hotspot[1], hotspot[0]], icon=folium.Icon(color='red')).add_to(map)

# Add a marker for the predicted crime location (using Option 2: next_hotspot from the previous answer)
folium.Marker([next_hotspot[1], next_hotspot[0]], icon=folium.Icon(color='green')).add_to(map)

# Save the map as an HTML file
map.save("crime_hotspots_and_prediction.html")
webbrowser.open("crime_hotspots_and_prediction.html")

import pandas as pd
from sklearn.cluster import DBSCAN

# Load dataset
data = pd.read_csv("training.csv")
data = data.dropna(subset=['Longitude', 'Latitude'])
locations = data[['Longitude', 'Latitude']]

# Set the DBSCAN parameters
eps = 0.01
min_samples = 10

# Create and fit the DBSCAN model
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(locations)

# Assign the cluster labels to your dataset
data['Cluster'] = dbscan.labels_

# Visualize the clusters
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=12)

# Assign a color to each cluster
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

for index, row in data.iterrows():
    color = colors[row['Cluster'] % len(colors)] if row['Cluster'] >= 0 else 'gray'
    location = (row['Latitude'], row['Longitude'])
    folium.CircleMarker(
        location=location,
        radius=5,
        popup=f"Cluster: {row['Cluster']}",
        fill=True,
        color=color,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(crime_map)

crime_map.save("crime_clusters.html")
webbrowser.open("crime_clusters.html")

# Generate synthetic crime data
data = pd.read_csv('training.csv')

data = data[data['Crime ID'] != '']
data = data[data['Latitude'].notnull()]
data = data[data['Longitude'].notnull()]
data = data[data['Crime type'].notnull()]
data = data[data['Longitude'] != 0]
data = data[data['Latitude'] != 0]
data = data.dropna(subset=['Longitude', 'Latitude'])
print("Number of rows in the dataset:", len(data))

# make a new dataframe with only the columns we want
data = data[['Latitude', 'Longitude', 'Crime type']]

