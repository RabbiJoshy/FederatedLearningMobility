import pandas as pd
import geopandas as gpd
import os
from Felyx.FelyxPredictions.FelyxUtils import MakeFelyxGeo
import pandas as pd

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