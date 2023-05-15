import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import geopandas as gpd
import datetime
#districts =gpd.read_file('amsregions.json')
districts =gpd.read_file('AmsterdamGeoJsons/AmsterdamGebieden.json')
grouped = pd.read_pickle(os.path.join('FelyxAnimations', 'data', '1T', 'DXH13K'))
grouped['change'] = grouped.lon.diff() + grouped.lat.diff() #Use .shift() to deal with non-numericsal
changed = grouped[abs(grouped['change']) > 0.001]

period = '30T'
groupedMar = pd.read_pickle(os.path.join('GeoData','Felyx','GroupedMar2023','P50' + period))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# tx = ax.set_title('Frame 0')

def animatefelyxonebike(i):
   ax.clear()
   tx = ax.set_title('Frame 0')
   districts.plot(ax=ax, facecolor="none",
                   edgecolor='black', lw=0.7, aspect = 1)
   subdf = df.iloc[i:i + 1, :]
   subdf.plot(ax=ax)

   tx.set_text('Felyx {0}'.format(str(df.index[i])))

   return

felyx = groupedMar.merge(districts, on = 'Gebied', how="inner")
felyxgeo = gpd.GeoDataFrame(
      felyx,
      geometry= felyx.geometry,
      crs='EPSG:4326'
   )

def FelyxRegions(felyxgeo):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    tx = ax.set_title('Frame 0')

    animationdf = felyxgeo.set_index('time')
    timerange2 = animationdf.index.unique().sort_values()

    def AmsInit():
        districts.plot(ax=ax, facecolor="none", edgecolor='black', lw=0.4, aspect = 1)
        return


    def animatefelyxstadsdeelen(i):
        timestamp = timerange2[i]
        ax.clear()
        tx = ax.set_title('Frame 0')
        # tx = ax.set_title(timestamp)
        slice = animationdf.loc[timestamp]
        districts.plot(ax=ax, facecolor="none", edgecolor='black', lw=0.4, aspect = 1)
        if i == 1:
            slice.plot(ax = ax, column = 'carId', cmap='Blues', vmin= 1, vmax= 8, legend = True)#2)
        else:
            slice.plot(ax=ax, column='carId', cmap='Blues', vmin=1, vmax=8)

        tx.set_text('Felyx {0}'.format(str(timestamp)))


        return

    ani = animation.FuncAnimation(fig, animatefelyxstadsdeelen, frames= 200, init_func=AmsInit) #len(timerange2)
    FFwriter = animation.FFMpegWriter()
    ani.save('gebieden.mp4', writer=FFwriter)


    return

FelyxRegions(felyxgeo)


# ani = animation.FuncAnimation(fig, animate, frames=len(df))
# FFwriter = animation.FFMpegWriter()
# ani.save('plot.mp4', writer=FFwriter)

ani = animation.FuncAnimation(fig, animatefelyxstadsdeelen, frames=200, init_func = AmsInit)#len(timerange2))
FFwriter = animation.FFMpegWriter()
ani.save('gebieden.mp4', writer=FFwriter)
view = sfg.sample(100)
