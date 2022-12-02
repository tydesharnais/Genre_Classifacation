from youtubesearchpython import VideosSearch
from pytube import YouTube
import os 
import pandas as pd
from pathlib import Path
from multiprocessing import Pool



def download_Video(url : str, songName : str):
        # url input from user
    cur_path = os.path.dirname(__file__)
    yt = YouTube(url)
    # extract only audio
    video = yt.streams.filter(only_audio=True).first()
    # download the file
    out_file = video.download(output_path=cur_path,filename=songName)
    # save the file
    base, ext = os.path.splitext(out_file)
    
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    # result of success
    print(yt.title + " has been successfully downloaded.")

def search_Video(query : str):
    default_YT_String = 'https://www.youtube.com/watch?v='

    videosSearch = VideosSearch(query, limit = 1)
    results = videosSearch.result()
    result1 = results['result']
    results_Dict = result1[0]
    
    return default_YT_String + results_Dict['id']

def main():
    df = pd.read_csv('songs_dataset_Youtube.csv')
    df = df.dropna()
    df = df.drop_duplicates()
    

    for i in range(2000):
        query = df.iloc[i][3]
        songName = df['Song'][i]
        artistName = df['Artist'][i]
        song_Artist = f'{songName}_{artistName}'
        url_String = search_Video(query)
        
        try:
            download_Video(url_String, song_Artist)
        except:
            if(os.path.exists(song_Artist+'.mp4')):
                print(f'{song_Artist}.mp4 exists.. Removing redundency')
                os.remove(song_Artist+'.mp4')
            continue
    print("\n\nCOMPLETED DOWNLOADS\n\n")


if __name__ == '__main__':
    main()


    

