import pandas as pd
import geopandas as gpd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
districts =gpd.read_file('PublicGeoJsons/AmsterdamGebieden.json')
# df = pd.read_pickle(os.path.join(
#     'CleanData', 'Felyx', '2023','03', 'month', 'Amsterdam50'))#.sample
# df = df[['carId', 'fuelLevel', 'time', 'lat', 'lon']]
# geodf = MakeFelyxGeo(df)
# geodf = geodf.sjoin(districts, how="inner", predicate='intersects')
# geodf = geodf[['carId', 'fuelLevel', 'time', 'geometry',
#        'Gebied', 'Stadsdeel']]
# geodf.to_pickle('GeoData/Felyx/Mar23_50')


df = pd.read_pickle('Change Data/movers')

# fig, ax = plt.subplots()
# districts.plot(ax = ax, facecolor = "none")
# df.plot(ax = ax, markersize = df.fuelLevel/10)
# plt.show()

modeldf = df.copy()
modeldf['lon'] = modeldf.prev_location.x
modeldf['lat'] = modeldf.prev_location.y

modeldf['seconds'] = modeldf['prev_time'].dt.second + 60*(modeldf['prev_time'].dt.hour)
seconds_in_day = 24*60*60
import numpy as np

modeldf['sin_time'] = np.sin(2*np.pi*modeldf.seconds/seconds_in_day)
modeldf['cos_time'] = np.cos(2*np.pi*modeldf.seconds/seconds_in_day)
modeldf.drop('seconds', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    modeldf[['prev_Gebied', 'lat', 'lon', 'prev_fl', 'sin_time', 'cos_time']],
    modeldf[['Stadsdeel']], test_size=0.33, random_state=42)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
X_train.iloc[:,0] = le.fit_transform(X_train.iloc[:,0])
X_test.iloc[:,0] = le.fit_transform(X_test.iloc[:,0])
y_test = le.fit_transform(y_test)


clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print(accuracy_score(y_test, y_pred))
predictionDF = pd.DataFrame({'test': le.inverse_transform(y_test), 'pred': le.inverse_transform(y_pred)})
predictionDF['pred'].value_counts()
len(df[df['Gebied'] == df['prev_Gebied']])/len(df)