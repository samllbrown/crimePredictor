import webbrowser

import folium
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from geopy import Nominatim


def get_color(crime_type):
    if ((type(crime_type) == str) != True):
        return 'black'
    if crime_type == 'Anti-social behaviour':
        return 'red'
    elif crime_type == 'Bicycle theft':
        return 'blue'
    elif crime_type == 'Burglary':
        return 'green'
    elif crime_type == 'Criminal damage and arson':
        return 'purple'
    elif crime_type == 'Drugs':
        return 'orange'
    elif crime_type == 'Other theft':
        return 'darkred'
    elif crime_type == 'Possession of weapons':
        return 'darkblue'
    elif crime_type == 'Public order':
        return 'darkgreen'
    elif crime_type == 'Robbery':
        return 'cadetblue'
    elif crime_type == 'Shoplifting':
        return 'darkpurple'
    elif crime_type == 'Theft from the person':
        return 'white'
    elif crime_type == 'Vehicle crime':
        return 'pink'
    elif crime_type == 'Violence and sexual offences':
        return 'beige'
    elif crime_type == 'Other crime':
        return 'lightred'

print("TensorFlow version:", tf.__version__)
df = pd.read_csv("southwales12.csv", delimiter=",", skip_blank_lines=True, skipinitialspace=True, engine='python',
                 header=0)
print(df.head(5))
df = df[df['Crime ID'] != '']
df = df.set_index("Crime ID")

locator = Nominatim(user_agent='myGeocoder')
location = locator.geocode('Swansea, Wales')

longitudes = df['Longitude'].tolist()
latitudes = df['Latitude'].tolist()
locations = list(zip(longitudes, latitudes))
print(df.head(5))

df = df[df['Longitude'] > -4.044780]
df = df[df['Longitude'] < -3.772868]
df = df[df['Latitude'] > 51.579037]
df = df[df['Latitude'] < 51.702400]
df = df[df['Latitude'].notnull()]
df = df[df['Longitude'].notnull()]
df = df[df['Crime type'].notnull()]

df.plot(kind='scatter', x='Longitude', y='Latitude', alpha=0.4, figsize=(10, 7), c='red', colorbar=True, sharex=False)
plt.show()

map1 = folium.Map(
    location=[51.624915, -3.940063],
    tiles='cartodbpositron',
    zoom_start=15,

)

for row in df.itertuples():
    crimeType = df['Crime type'][row.Index]
    if('Crime ID' in crimeType):
        crimeType = 'Other crime'
    folium.CircleMarker(
        location=[row.Latitude, row.Longitude],
        radius=5,
        color=get_color(crimeType),
        fill=True,
        fill_color=get_color(crimeType),
        fill_opacity=0.7,
        popup=crimeType,
    ).add_to(map1)

legend_html =   '''
                <div style="position: fixed;
                            bottom: 50px; left: 50px; width: 250px; height: 450px;
                            border:2px solid grey; z-index:9999; font-size:14px;
                            "><b>&nbsp; Crime Type &nbsp; </b><br>
                                &nbsp; Anti-social behaviour &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                                &nbsp; Bicycle theft &nbsp; <i class="fa fa-map-marker fa-2x" style="color:blue"></i><br>
                                &nbsp; Burglary &nbsp; <i class="fa fa-map-marker fa-2x" style="color:green"></i><br>
                                &nbsp; Criminal damage and arson &nbsp; <i class="fa fa-map-marker fa-2x" style="color:purple"></i><br>
                                &nbsp; Drugs &nbsp; <i class="fa fa-map-marker fa-2x" style="color:orange"></i><br>
                                &nbsp; Other theft &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkred"></i><br>
                                &nbsp; Possession of weapons &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkblue"></i><br>
                                &nbsp; Public order &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkgreen"></i><br>
                                &nbsp; Robbery &nbsp; <i class="fa fa-map-marker fa-2x" style="color:cadetblue"></i><br>
                                &nbsp; Shoplifting &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkpurple"></i><br>
                                &nbsp; Theft from the person &nbsp; <i class="fa fa-map-marker fa-2x" style="color:white"></i><br>
                                &nbsp; Vehicle crime &nbsp; <i class="fa fa-map-marker fa-2x" style="color:pink"></i><br>
                                &nbsp; Violence and sexual offences &nbsp; <i class="fa fa-map-marker fa-2x" style="color:beige"></i><br>
                                &nbsp; Other crime &nbsp; <i class="fa fa-map-marker fa-2x" style="color:lightred"></i><br>
                </div>
                '''

map1.get_root().html.add_child(folium.Element(legend_html))
map1.save('map1.html')
#webbrowser.open('map1.html')

df = pd.read_csv("training.csv", delimiter=",", skip_blank_lines=True, skipinitialspace=True, engine='python',
                 header=0)
df = df[df['Crime ID'] != '']
df = df.set_index("Crime ID")
df = df[df['Longitude'] > -4.044780]
df = df[df['Longitude'] < -3.772868]
df = df[df['Latitude'] > 51.579037]
df = df[df['Latitude'] < 51.702400]
df = df[df['Latitude'].notnull()]
df = df[df['Longitude'].notnull()]
df = df[df['Crime type'].notnull()]

# make a model that uses the read in data
tensorflow = tf.keras.models.Sequential()
tensorflow.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
tensorflow.add(tf.keras.layers.Dense(2))
tensorflow.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# train the model
tensorflow.fit(df[['Longitude', 'Latitude']], df[['Longitude', 'Latitude']], epochs=5)
# save the model
tensorflow.save('model.h5')

tensorflow = tf.keras.models.load_model('model.h5')
df = pd.read_csv("training.csv", delimiter=",", skip_blank_lines=True, skipinitialspace=True, engine='python', header=0)
df = df[df['Crime ID'] != '']
df = df.set_index("Crime ID")
df = df[df['Longitude'] > -4.044780]
df = df[df['Longitude'] < -3.772868]
df = df[df['Latitude'] > 51.579037]
df = df[df['Latitude'] < 51.702400]
df = df[df['Latitude'].notnull()]
df = df[df['Longitude'].notnull()]
df = df[df['Crime type'].notnull()]
df = df[['Longitude', 'Latitude']]

predictions = tensorflow.predict(df)
plt.scatter(predictions[:, 0], predictions[:, 1], c='r')
plt.show()











