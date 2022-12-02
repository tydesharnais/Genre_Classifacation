import pandas as pd
import os
from pathlib import Path

def load_Data():
    cur_path = os.path.dirname(__file__)
    file_path = Path(cur_path+'\data\spotifydata\songs_normalize.csv')
    
    if(os.path.exists(file_path)):
        df = pd.read_csv(file_path)
        return df
    else:
        print('Error in finding CSV file')
        exit()

def handle_nulls(df):

    df = df.dropna()
    return df

def set_index(df):
    df = df.set_index(df['song'])
    return df

def change_dtypes(df):
    #artist,song,duration_ms,explicit,year,popularity,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,genre
    # change explicit column to int
    df['explicit'] = df.explicit.astype('int')
    df['has_feat'] = df.has_feat.astype('int')
    #df['disc_number'] = df.disc_number.astype('int')
    df['mode'] = df['mode'].astype('int')
    df['key'] = df.key.astype('int')
    df['duration_seconds'] = df.duration_seconds.astype('int')
    df['duration_minutes'] = df.duration_minutes.astype('int')
    df['duration_ms'] = df.duration_ms.astype('int')
    df['popularity'] = df.popularity.astype('int')
    df['year'] = df.year.astype('int')

    return df

def prepare_df(df):
    
    df = handle_nulls(df)
    df = change_dtypes(df)
    
    return df