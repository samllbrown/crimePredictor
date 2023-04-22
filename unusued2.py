import pandas as pd
import matplotlib.pyplot as plt

tr = pd.read_csv("training.csv", delimiter=",", skip_blank_lines=True, skipinitialspace=True, engine='python', header=0)
tr = tr[tr['Crime ID'] != '']
tr = tr.set_index("Crime ID")
tr = tr[tr['Longitude'] > -4.044780]
tr = tr[tr['Longitude'] < -3.772868]
tr = tr[tr['Latitude'] > 51.579037]
tr = tr[tr['Latitude'] < 51.702400]
tr = tr[tr['Latitude'].notnull()]
tr = tr[tr['Longitude'].notnull()]
tr = tr[tr['Crime type'].notnull()]

df = pd.read_csv("southwales12.csv", delimiter=",", skip_blank_lines=True, skipinitialspace=True, engine='python',
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

# normalise the longitude and latitude values of df
df['Longitude'] = (df['Longitude'] - df['Longitude'].min()) / (df['Longitude'].max() - df['Longitude'].min())
df['Latitude'] = (df['Latitude'] - df['Latitude'].min()) / (df['Latitude'].max() - df['Latitude'].min())
# print all the latitudes and longitudes in the south wales dataset
plt.scatter(df['Longitude'], df['Latitude'], c='r')

# normalise the longitude and latitude values
tr['Longitude'] = (tr['Longitude'] - tr['Longitude'].min()) / (tr['Longitude'].max() - tr['Longitude'].min())
tr['Latitude'] = (tr['Latitude'] - tr['Latitude'].min()) / (tr['Latitude'].max() - tr['Latitude'].min())
# print all the latitudes and longitudes in the training set
plt.scatter(tr['Longitude'], tr['Latitude'], c='b')
#plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train, test = train_test_split(tr, test_size=0.2)
train, val = train_test_split(train, test_size=0.1)


model = LinearRegression()
model.fit(train[['Longitude', 'Latitude']], train[['Longitude', 'Latitude']])
print(model.score(test[['Longitude', 'Latitude']], test[['Longitude', 'Latitude']]))
predictions = model.predict(test[['Longitude', 'Latitude']])
plt.scatter(predictions[:, 0], predictions[:, 1], c='r')
#plt.show()

import folium
import webbrowser




map1 = folium.Map(
    location=[51.624915, -3.940063],
    tiles='cartodbpositron',
    zoom_start=15,

)
df = pd.read_csv("southwales12.csv", delimiter=",", skip_blank_lines=True, skipinitialspace=True, engine='python',
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
webbrowser.open('map1.html')



# make a model using knn regression to predict the longitude and latitude of a crime, then find the optimal k value, and the optimal weights
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# dictionary of parameters to test
params = {'n_neighbors': [1, 5, 10, 100, 200, 400, 600, 800, 1000], 'weights': ['uniform', 'distance']}
knn = KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
# fit the model to the training data
model.fit(train[['Longitude', 'Latitude']], train[['Longitude', 'Latitude']])

# print each parameter and its corresponding score in order of distance or uniform
for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
    print(param, score)



# show both the linear regression and knn regression on the same graph
plt.scatter(predictions[:, 0], predictions[:, 1], c='r')
plt.scatter(model.predict(test[['Longitude', 'Latitude']])[:, 0], model.predict(test[['Longitude', 'Latitude']])[:, 1], c='b')
#plt.show()


# make a model using random forest regression to predict the longitude and latitude of a crime
from sklearn.ensemble import RandomForestRegressor

# dictionary of parameters to test
params = {'n_estimators': [1, 5, 10, 100, 200, 400, 600, 800, 1000], 'max_depth': [1, 5, 10, 100, 200, 400, 600, 800, 1000]}
rf = RandomForestRegressor()
model = GridSearchCV(rf, params, cv=5)
model.fit(train[['Longitude', 'Latitude']], train[['Longitude', 'Latitude']])

print(model.score(test[['Longitude', 'Latitude']], test[['Longitude', 'Latitude']]))
print(model.best_params_)
predictions = model.predict(df[['Longitude', 'Latitude']])
print(model.score(df[['Longitude', 'Latitude']], df[['Longitude', 'Latitude']]))

plt.scatter(predictions[:, 0], predictions[:, 1], c='r')
plt.show()

















