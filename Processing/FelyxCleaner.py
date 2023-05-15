import pandas as pd
import random
import tarfile
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
day = '2023-03-03'
year = '2023'
month = '03'

directory = 'L:\\UserData\\Joshua'#\\2023\\03\\2023-03-07.tar.xz'
daysinmonth = os.listdir(os.path.join(directory, 'data', 'felyx', year, month))
def CollateFelyxDay(directory, day, save = True):
    path = os.path.join(directory, 'data', 'felyx', day[:4], day[5:7], day)# + '.tar.xz')
    dflist = []
    tar = tarfile.open(os.path.join(path), 'r:xz')
    members = tar.getmembers()
    # subset_of_times = random.sample(members, 5)

    for subfilename in members[1:]:
        # if subfilename.name.endswith('xz'):
        # print(subfilename.name)
        data = tar.extractfile(subfilename)
        event_df = pd.read_json(data)
        event_df['time'] = pd.to_datetime(subfilename.name[11:21], unit='s')
        dflist.append(event_df)
    df = pd.concat(dflist)
    df = df.drop(['serviceType', 'title', 'vehicleStateId',
                  'reservationState', 'distance', 'address'], axis = 1)

    if save == True:
        os.makedirs(os.path.join('CleanData/Felyx', day[:4], day[5:7]), exist_ok=True)
        df.to_pickle(os.path.join('CleanData/Felyx', day[:4], day[5:7], day[8:10]))

    return df
# df = CollateFelyxDay(directory, day)

def CollateFelyxDaysInMonth(year, month):
    daysinmonth = os.listdir(os.path.join(directory, 'data', 'felyx', year, month))
    for days in daysinmonth[17:19]:
        if days.endswith('xz'):
            print(days)
            CollateFelyxDay(directory, days)

    return
# CollateFelyxDaysInMonth(year, month)


def CollateFelyxMonth(directory, year, month, save = True ,city = None):
    path = os.path.join('CleanData', 'Felyx', year, month)
    li = []
    for filename in os.listdir(path):
        if not os.path.isdir(os.path.join(path, filename)):
            print(filename)
            df = pd.read_pickle(os.path.join(path, filename))
            if city:
                df = df[df['city'] == city]
            li.append(df)
    #
    df = pd.concat(li, axis=0, ignore_index=True)

    if save == True:
        os.makedirs(os.path.join(path, 'month'), exist_ok = True)

        if city:
            outfile = os.path.join(path, 'month', city)
        else:
            outfile = os.path.join(path, 'month', 'all')
        df.to_pickle(outfile)

    return df
collate = CollateFelyxMonth(directory, '2023', '03', city = 'Amsterdam')

def afewplates(directory, year, month, city = 'Amsterdam', nplates = 25):
    path = os.path.join('CleanData', 'Felyx', year, month, 'month')
    subdf = pd.read_pickle(os.path.join(path, city))
    plates = subdf['licencePlate'].unique()
    subdf = subdf[subdf['licencePlate'].isin(plates[:nplates])]

    subdf.to_pickle(os.path.join(path, city+ str(nplates)))
    return subdf
subb = afewplates(directory, '2023', '03', city = 'Amsterdam')