import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os

Gebieden =gpd.read_file('PublicGeoJsons/AmsterdamGebieden.json')
Stadsdelen = gpd.read_file('PublicGeoJsons/AmsterdamStadsdelen.json')
df = pd.read_pickle('NSMarchsorted')[:20000]
geodf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs='EPSG:4326'
)
geodf.geometry.crs = "EPSG:4326"
geodf.to_crs('EPSG:4326')

change = geodf[geodf['num_bikes_available'] != geodf['num_bikes_available'].shift()]
change2 = change[change['station_id'] == change['station_id'].shift()]
change2['diffs'] = change2['num_bikes_available'].diff()/ np.sqrt(change2['num_bikes_available'])
change3 = change2[abs(change2['diffs']) < 1]

Movements_NS_Amsterdam = change3.sjoin(Gebieden, how="inner", predicate='intersects')
Movements_NS_Amsterdam.to_pickle(os.path.join('Change Data','Movements_NS_Amsterdam'))

def bikes_in_number(df):
    totalin = dict()
    for i in df['time'].unique():
        # ex = geodf['time'].iloc[i]
        totalin[i] = df[df['time'] == i]['num_bikes_available'].sum()
    return totalin



amsonly = geodf.sjoin(Gebieden, how="inner", predicate='intersects')
amsonlypart = amsonly[:100000]

totalb = bikes_in_number(amsonly)
totalbdf = pd.Series(totalb)

totalbdf.to_pickle('availibility')




import requests
a = requests.get("https://opendata.cbs.nl/ODataApi/OData/83505NED/")



