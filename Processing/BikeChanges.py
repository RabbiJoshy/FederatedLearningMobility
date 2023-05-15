import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
ams = pd.read_json('AmsterdamGeoJsons/provinces.geojson')
utr = gpd.read_file('wijken-utrecht.geojson')
df = pd.read_csv('CleanData/GoAbout/0307/Day/0307')
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs='EPSG:4326'
)
Subdistrict = 'Gebieden'
districts =gpd.read_file('AmsterdamGeoJsons/AmsterdamGebieden.json')

UTGA = gdf[(gdf.lat.between(52,52.2)) & (gdf.lon.between(5,5.2)) ] #Utrechtforga
UTGA = UTGA[['num_bikes_available', 'station_id',
       'time', 'name', 'geometry']]

ams = gdf[gdf['city'] == 'am']
amsgeo = ams.sjoin(districts, how="inner", predicate='intersects')
# amsgeo = amsgeo[['station_id', 'time_x', 'geometry', 'num_bikes_available', 'Gebied']]
# ams10 = amsgeo[:1000]
# ams10 = ams10.set_index(ams10.time_x).sort_index()
# st = ams10.groupby('station_id')['num_bikes_available'].shift()
# amsgeo['delta'] = delta

ams = UTGA.set_index(ams.time).sort_index()

grouped = ams.groupby('station_id')['num_bikes_available'].shift()

ams['prev'] = grouped

delta = ams[ams.prev != ams.num_bikes_available]
delta = delta.sort_values(['station_id', 'time'])

# donk = ams.groupby('station_id').sort_values('time')
# donk = ams.sort_values(['station_id', 'time'])


change = donk[donk['num_bikes_available'] != donk['num_bikes_available'].shift()]
change2 = change[change['station_id'] == change['station_id'].shift()]

fig, ax = plt.subplots()
utr.plot(ax=ax, facecolor="none", edgecolor='black', lw=0.4, aspect=1)
ams.plot(ax=ax, column='num_bikes_available', legend=True, cmap='Blues')
plt.show()
ax.clear()

statiosn = ams['station_id'].unique()

ams.groupby('station_id').nth(0).plot()

view = ams[:10000][['num_bikes_available', 'station_id',
       'time', 'name', 'geometry']]
