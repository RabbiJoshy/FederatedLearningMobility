import pandas as pd
import geopandas as gpd
import os
from Felyx.FelyxPredictions.FelyxUtils import MakeFelyxGeo
import pandas as pd
import matplotlib.pyplot as plt

Subdistrict = 'Gebieden'


districts =gpd.read_file('AmsterdamGeoJsons/AmsterdamGebieden.json')
# df = pd.read_pickle(os.path.join(
#     'CleanData', 'Felyx', '2023','03', 'month', 'Amsterdam50'))#.sample

df = pd.read_pickle(os.path.join(
    'CleanData', 'Felyx', '2023','03', '10'))#.sample

df = df[['carId', 'fuelLevel', 'time', 'lat', 'lon']]
geodf = MakeFelyxGeo(df)
geodf = geodf.sjoin(districts, how="inner", predicate='intersects')
geodf = geodf[['carId', 'fuelLevel', 'time', 'geometry',
       'Gebied', 'Stadsdeel']]
geodf.to_pickle('GeoData/Felyx/20230310' + Subdistrict)





Subdistrict = 'Gebieden'
df = pd.read_pickle('GeoData/Felyx/20230310' + Subdistrict)

plates = df['carId'].unique()

df['prev_location'] = df.groupby('carId')['geometry'].shift()
df['prev_Gebied'] = df.groupby('carId')['Gebied'].shift()
df['prev_time'] = df.groupby('carId')['time'].shift()
df['prev_fl'] = df.groupby('carId')['fuelLevel'].shift()


sub = df
sub['movement'] = sub['Gebied']+ sub['prev_Gebied']
sub['distancemoved'] = sub.geometry.distance(sub.prev_location)

movers = sub[sub['distancemoved'] > 0.001]
plates2 = movers['carId'].unique()
movers.to_pickle('movers')


from collections import Counter
mov = Counter(movers.movement)
movdf = pd.DataFrame(mov)

movdf = movers.groupby(['prev_Gebied', 'Gebied']).size().rename('movements').reset_index()


for gebied in movdf['Gebied'].unique()[:]:
    fig, ax = plt.subplots()
    ax.set_title(gebied)
    DPR = movdf[movdf['Gebied'] == gebied]

    CW = DPR.merge(districts, left_on = 'prev_Gebied', right_on = 'Gebied')
    geodf  = gpd.GeoDataFrame(
            CW,
            geometry = CW.geometry,
            crs='EPSG:4326'
        )
    geodf['movementsnorm'] = geodf['movements'] / geodf['movements'].sum()
    districts.plot(ax=ax, facecolor="none", edgecolor='black', lw=0.4, aspect = 1)
    geodf.plot(ax = ax, column = 'movementsnorm', legend = True, cmap = 'Blues')
    plt.show()
    ax.clear()

geodf['movementsnorm'] = geodf['movements'] / geodf['movements'].sum()