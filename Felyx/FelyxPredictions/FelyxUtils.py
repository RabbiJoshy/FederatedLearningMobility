import pandas as pd
import geopandas as gpd

def MakeFelyxGeo(df):
    felyxgeo = gpd.GeoDataFrame(
        df,
        geometry = gpd.points_from_xy(df.lon, df.lat),
        crs='EPSG:4326'
    )
    felyxgeo.geometry.crs = "EPSG:4326"
    felyxgeo.to_crs('EPSG:4326')

    return felyxgeo