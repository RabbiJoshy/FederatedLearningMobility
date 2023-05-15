import pandas as pd
import json
import tarfile
import os
day = '0307'

def fixjson(file, addition = '', replace_gen = True, replace_quotes = True):

    # file = open(filepath, mode='r')
    txt = file.read()
    txt = txt.decode("utf-8")
    if replace_gen == True:
        # print('replace_gen')
        txt = txt.replace("\'t ", "het")
        txt = txt.replace("s\' ", "ss ")
        txt = txt.replace("\'s\"", "s\"")
        txt = txt.replace("\'s ", "s ")
        # txt = txt.replace("'S ", "SS ")
        # txt = txt.replace("'S- ", "SS ")
        # txt = txt.replace("S- ", "dffef ")
        # txt = txt.replace("\'s ", "SS ")
        # txt = txt.replace("\"S ", "SS ")
        txt = txt.replace('\"S-', 'SS-')
        txt = txt.replace('\'S-', 'SS-')
        txt = txt.replace('\"s-', 'ss-')
        txt = txt.replace('\'s-', 'ss-')
        # txt = txt.replace('\"SS-', 'SS-')
    if replace_quotes == True:
        # print('RQ')
        txt = txt.replace("\'1.1\'", "1.1")
        txt = txt.replace("\'1.", "1")
        txt = txt.replace("C\'h", "Ch")
        txt = txt.replace("\"Midget Golf\"", "Midget Golf")
        txt = txt.replace("\"Amerika\"", "Amerika")
        txt = txt.replace("\'", "\"")
    # import unidecode
    # txt = unidecode.unidecode(txt)
    txt = txt.replace(u'\\xa0', u' ')
    cleanjson = txt.replace("True", "true")
    # print('cleaned')
    cleanjson += addition
    jsondict = json.loads(cleanjson)

    return jsondict
def make_geo(df, EPSG = "EPSG:4326"):
    GeoDF = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs=EPSG
    )
    GeoDF.geometry.crs = EPSG
    GeoDF.to_crs(EPSG)

    GeoDF = gpd.GeoDataFrame(GeoDF, geometry='geometry')

    return GeoDF

def Donkey(path):
    df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.startswith('donkey'):
            filenameinfo = filename.replace(".tar.gz", "")
            timestamp = filenameinfo[-12:]

            tar = tarfile.open(os.path.join(path, filename), 'r:gz')
            members = tar.getmembers()
            for subfilename in members:

                if subfilename.name == 'station_status':
                    data = fixjson(tar.extractfile(subfilename))['data']['stations']
                    subdf = pd.DataFrame.from_dict(data, orient='columns')

                    time = pd.Timestamp(year=int(timestamp[:4]), month=int(timestamp[4:6]), day=int(timestamp[6:8]), hour=int(timestamp[8:10]), minute= int(timestamp[10:12]))
                    subdf['city'] = filenameinfo[7:-23]
                    subdf['time'] = time
                    # print(subdf)
                    # df = pd.concat([subdf, df])

                elif subfilename.name == 'station_information':
                    infodata = fixjson(tar.extractfile(subfilename),addition = '}')['data']['stations']
                    infosubdf = pd.DataFrame.from_dict(infodata, orient='columns')
                    # timestamp = filenameinfo[-12:]
                    # infosubdf['time'] = timestamp

            combined = pd.merge(subdf, infosubdf, on='station_id')
            df = pd.concat([combined, df])

    # GeoDF = make_geo(df)
    # Geo_DF = gpd.GeoDataFrame(GeoDF, geometry='geometry')
    #
    os.makedirs('CleanData/Donkey/0307', exist_ok=True)
    df.to_csv(os.path.join('CleanData/Donkey/0307', path[-2:]))

    return df
DonTest = Donkey(os.path.join('Bike '+ day, '03'))

def CKL(path):
    df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.startswith('CKL'):
            # print(filename)
            filenameinfo = filename.replace(".tar.gz", "")
            timestamp = filenameinfo[-12:]

            tar = tarfile.open(os.path.join(path, filename), 'r:gz')
            members = tar.getmembers()
            for subfilename in members:

                if subfilename.name == 'free_bike_status':
                    data = fixjson(tar.extractfile(subfilename))['data']['bikes']
                    subdf = pd.DataFrame.from_dict(data, orient='columns')

                    time = pd.Timestamp(year=int(timestamp[:4]), month=int(timestamp[4:6]), day=int(timestamp[6:8]), hour=int(timestamp[8:10]), minute= int(timestamp[10:12]))
                    subdf['time'] = time
                    # print(subdf)
                    df = pd.concat([subdf, df])


    # GeoDF = make_geo(df)
    # Geo_DF = gpd.GeoDataFrame(GeoDF, geometry='geometry')
    #
    os.makedirs('CleanData/CKL/0307', exist_ok=True)
    df.to_csv(os.path.join('CleanData/CKL/0307', path[-2:]))

    return df
CKLTest = CKL(os.path.join('Bike '+ day, '03'))

def GoAbout(path):
    """Station_information requires a change to the end after Version"""
    df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.startswith('goabout'):
            # print(filename)
            filenameinfo = filename.replace(".tar.gz", "")
            timestamp = filenameinfo[-12:]

            tar = tarfile.open(os.path.join(path, filename), 'r:gz')
            members = tar.getmembers()
            for subfilename in members:

                if subfilename.name == 'station_status':
                    data = fixjson(tar.extractfile(subfilename))['data']['stations']
                    subdf = pd.DataFrame.from_dict(data, orient='columns')

                    time = pd.Timestamp(year=int(timestamp[:4]), month=int(timestamp[4:6]), day=int(timestamp[6:8]), hour=int(timestamp[8:10]), minute= int(timestamp[10:12]))
                    subdf['time'] = time
                    # print(subdf)
                    # df = pd.concat([subdf, df])

                elif subfilename.name == 'station_information':
                    # print(filename)
                    infodata = fixjson(tar.extractfile(subfilename),addition = '"Unknown"}')['data']['stations']
                    # subfilepath = os.path.join(path, filename, subfilename)
                    # infodata = fixjson(subfilepath, addition = '}')['data']['stations']
                    infosubdf = pd.DataFrame.from_dict(infodata, orient='columns')
                    timestamp = filenameinfo[-12:]
                    # infosubdf['time'] = timestamp

            combined = pd.merge(subdf, infosubdf, on='station_id')
            df = pd.concat([combined, df])

    os.makedirs('CleanData/GoAbout/0307', exist_ok=True)
    df.to_csv(os.path.join('CleanData/GoAbout/0307', path[-2:]))

    return df
GA = GoAbout(os.path.join('Bike '+ day, '03'))

def clean_df_by_hour(day, operator):
    for hour in os.listdir(os.path.join('Bike '+ day)):
        if hour != '.DS_Store':
            print(hour)
            path = os.path.join('Bike ' + day, hour)
            df = pd.DataFrame()
            if operator == 'Donkey':
                df = Donkey(path)
            elif operator == 'CKL':
                df = CKL(path)
            elif operator == 'GoAbout':
                df = GoAbout(path)
            else:
                print('Operator does not exist')

    return df
def collate_clean_day(day, operator):
    daydf = pd.DataFrame()
    for hour in os.listdir(os.path.join('CleanData', operator, day)):
        if hour != '.DS_Store':
            print(hour)
            hourdf = pd.read_csv(os.path.join('CleanData', operator, day, hour))
            daydf = pd.concat([daydf, hourdf])

    os.makedirs(os.path.join('CleanData', operator, day, 'Day'), exist_ok= True)
    daydf.to_csv(os.path.join('CleanData', operator, day, 'Day', day))

    return daydf

hou = clean_df_by_hour(day, 'GoAbout')
daaa = collate_clean_day(day, 'GoAbout')