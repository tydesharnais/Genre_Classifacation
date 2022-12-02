# Usual Libraries
import pandas as pd
from multiprocessing import Pool
import functools
from time import sleep
import os
from time import perf_counter
import math
# Librosa (the mother of audio files)
import librosa
import librosa.display
import warnings
import numpy as np
import json
warnings.filterwarnings('ignore')

#Custom
from prepare import load_Data

def asci_Art():
    print("""  
     _____                           _   _             
    /  __ \                         | | (_)            
    | /  \/ ___  _ ____   _____ _ __| |_ _ _ __   __ _ 
    | |    / _ \| '_ \ \ / / _ \ '__| __| | '_ \ / _` |
    | \__/\ (_) | | | \ V /  __/ |  | |_| | | | | (_| |
    \____/\___/|_| |_|\_/ \___|_|   \__|_|_| |_|\__, |
                                                _ _/ |
                                                |___/ """)

#If you want a spectogram image

def get_Spectogram(audio_file, sr, song_Name, artist_Name):
    # # Default FFT window size
    # n_fft = 2048 # FFT window size
    # hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

    # # Short-time Fourier transform (STFT)
    # D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))

    # # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    # DB = librosa.amplitude_to_db(D, ref = np.max)

    # # Creating the Spectogram
    # plt.figure(figsize = (16, 6))
    # librosa.display.specshow(DB, sr = sr, hop_length = hop_length,
    #                         cmap = 'cool')
    # plt.colorbar()
    
    # save_Path = f'{os.getcwd()}/Spectograms/{song_Name}_{artist_Name}.png'
    # plt.savefig(save_Path)

    return 0

# CALL THIS IN if 'name' == main(). 
# This is for generating time-series data for each song via segments in mfccs
# #
def get_MFCC_array_dataset():

    #VARS for MFCC generation via time-step / frequency
    num_Segments = 5
    hop_length = 512
    n_fft = 2048
    Sample_Rate = 22050
    Duration = 60

    samples_per_track = Sample_Rate* Duration
    num_samples_Seg = int(samples_per_track / num_Segments)
    
    expected_MFCC_vectors = math.ceil(num_samples_Seg / hop_length)

    df = pd.read_csv('final_dataframe.csv')
    home_Path = os.getcwd()
    path = f'{os.getcwd()}/wav'
    os.chdir(path)

    df = df.drop_duplicates(subset=['song', 'artist'])
    df['song'] = df['song'].str.replace('\(', '')
    df['song'] = df['song'].str.replace('\)', '')
    df['genre'] = df['genre'].astype('string')
    df['mfcc_Array'] = [np.array([]) for _ in range(len(df))]
    
    
    for song in os.listdir():
        if song.endswith(".wav"):
            split_Song = song.split('_')
            artist = split_Song[1].split('.')
            opt = find_in_DF(df,str(split_Song[0]),str(artist[0]))

            if(opt[0] == True):         
                #Import the song 
            
                y, sr = librosa.load(song, sr=22050)
                audio_file, _ = librosa.effects.trim(y)
                print(f'Processing {song}')
                for s in range(num_Segments):
                    mfccs_list = []
                    start_Sample = num_samples_Seg * s
                    finish = start_Sample + num_samples_Seg
                    #store mfcc for segment if has expected length 
                    
                    mfccs = librosa.feature.mfcc(audio_file[start_Sample:finish], sr=sr,
                                                n_mfcc=20, hop_length=hop_length, n_fft=n_fft)
                    mfccs = mfccs.T #transfers
                    if len(mfccs) == expected_MFCC_vectors:
                        print(f'{song} Segment {s}')
                        print('Mfcc shape ')
                        print(mfccs.shape)
                        print(mfccs.shape[0])
                        sleep(20)
                        mfccs_list.append(mfccs.tolist())
                    else:
                        print('Incorrect Length')
                    if s==4:
                        nummy = np.array(mfccs_list)
                        print('Numpy Shape')
                        print(nummy.shape)
                        
                        #print(mfccs_list)
                        df.at[opt[1],'mfcc_Array'] = nummy
                        df.to_csv(f'{home_Path}/test.csv')
                        
    #Save the final dataframe 
    df.to_csv(f'{home_Path}/final_dataframe_with_mfccs.csv')
                        
#
# MULTIPROCESSED FUNCTIONS 
# #
def get_MFCCs(audio_file, sr):
    mfccs = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=20)
    return mfccs

def rolloff(audio_file, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)
    return (spectral_rolloff.mean(), spectral_rolloff.var())

def spectral_bandwidth(audio_file, sr):
    bandwidth = librosa.feature.spectral_bandwidth(audio_file, sr)
    return (bandwidth.mean(), bandwidth.var())

def spectral_centroid(audio_file, sr):
    spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)
    return (spectral_centroids.mean(), spectral_centroids.var())

def rms(audio_file):
    rms = librosa.feature.rms(audio_file)
    return (rms.mean(), rms.var())

def harmonics_percept(audio_file):
    y_harm, y_perc = librosa.effects.hpss(audio_file)
    return (y_harm.mean(), y_harm.var(), y_perc.mean(), y_perc.var())

def zero_Crossing(audio_file):
    zero_crossings = librosa.zero_crossings(audio_file, pad=False)
    return (zero_crossings.mean(), zero_crossings.var())

def mel_Spectogram(y, sr):
    Mel = librosa.feature.melspectrogram(y, sr)
    return (Mel.mean(),Mel.var())

def get_Chroma_stft_mean_var(audio_file, sr):
    hop_length = 5000
    # Chromogram
    chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
    return (chromagram.mean(),chromagram.var())

#
# Find Song in dataset
# #
def find_in_DF(df : pd.DataFrame, song_Name : str, song_Art : str):
    #Try to find song by name
    i=0
    text = ''
    inx_Number = 999
    for i in range(len(song_Name)):
        # print(f'{i}: len {len(song_Name)}')
        # print(text)
        #print(df.song.str.contains(text))
        songFlag = False
        new_Df= df[df.song.str.contains(text,case=False ,regex= False, na=False)]
        if (len(new_Df) == 1):
            #print('Found Song!')
            for ind in new_Df.index:
                inx_Number = ind
            songFlag = True
            break
        else:
            if(i != len(song_Name)):
                text = text+song_Name[i]
                i=i+1
            if(i == len(song_Name)):
                #song = text
                text = text + ' '
                artist_Text = ''
                j=0  
                for j in range(len(song_Art)):
                    #print(f'{j}: len {len(song_Art)}')
                    Artist_DF = new_Df[new_Df.artist.str.contains(artist_Text,case=False ,regex= True, na=False)]
                    # print(f'Text: {text}')
                    # print(f'Artist Text: {artist_Text}')
                    if (len(Artist_DF) == 1):
                        #print("FOUND")
                        songFlag = True
                        #Artist_DF.head()
                        for ind in Artist_DF.index:
                            inx_Number = ind
                        #print(f'INDEX CHECK {inx_Number}')
                        break
                    
                    if(j == len(song_Art)-1):
                        #print(f'FINAL Artist Text: {artist_Text}')
                        #print(Artist_DF.head())
                        Artist_DF = Artist_DF[Artist_DF.artist.str.contains(artist_Text,case=False ,regex=False, na=False)]
                        if (len(Artist_DF) == 1):
                            #print("FOUND")
                            for ind in Artist_DF.index:
                                inx_Number = ind
                            songFlag = True
                            break
                        elif(len(Artist_DF) > 1):
                            for ind in Artist_DF.index:
                                exact_Match_song = Artist_DF[Artist_DF['song'].str.fullmatch(song_Name,case=False)]
                                if not exact_Match_song.empty:
                                    exact_Match_artist = exact_Match_song[exact_Match_song['artist'].str.fullmatch(song_Art,case=False)]
                                    if not exact_Match_artist.empty:
                                        if len(exact_Match_artist) == 1:
                                            #print("FOUND")
                                            for ind in Artist_DF.index:
                                                inx_Number = ind
                                            songFlag = True
                                            break
                        else: 
                            songFlag = False
                    else:
                        artist_Text = artist_Text + song_Art[j]
    if(songFlag == True):
        return (songFlag,inx_Number)
    else:
        #print(f'--------\nCOULDNOT FIND SONG {song_Name} by {song_Art}\n----------------')
        songFlag = False
        return (songFlag,inx_Number)

def smap(f):
    return f()

def main():

    df = load_Data()
    df['rolloff-mean'] = float(0)
    df['rolloff-var'] = float(0)
    df['bandwidth-mean'] = float(0)
    df['bandwidth-var'] = float(0)
    df['centroid-mean'] = float(0)
    df['centroid-var'] = float(0)
    df['rms-mean'] = float(0)
    df['rms-var'] = float(0)
    df['harmonic-mean'] = float(0)
    df['harmonic-var'] = float(0)
    df['percpt-mean'] = float(0)
    df['percpt-var'] = float(0)
    df['zeroCross-mean'] = float(0)
    df['zeroCross-var'] = float(0)
    df['mel-mean'] = float(0)
    df['mel-var'] = float(0)
    df['chroma-mean'] = float(0)
    df['chroma-var'] = float(0)
    df['mfcc1-mean'] = float(0)
    df['mfcc1-var'] = float(0)
    df['mfcc2-mean'] = float(0)
    df['mfcc2-var'] = float(0)
    df['mfcc3-mean'] = float(0)
    df['mfcc3-var'] = float(0)
    df['mfcc4-mean'] = float(0)
    df['mfcc4-var'] = float(0)
    df['mfcc5-mean'] = float(0)
    df['mfcc5-var'] = float(0)
    df['mfcc6-mean'] = float(0)
    df['mfcc6-var'] = float(0)
    df['mfcc7-mean'] = float(0)
    df['mfcc7-var'] = float(0)
    df['mfcc8-mean'] = float(0)
    df['mfcc8-var'] = float(0)
    df['mfcc9-mean'] = float(0)
    df['mfcc9-var'] = float(0)
    df['mfcc10-mean'] = float(0)
    df['mfcc10-var'] = float(0)
    df['mfcc11-mean'] = float(0)
    df['mfcc11-var'] = float(0)
    df['mfcc12-mean'] = float(0)
    df['mfcc12-var'] = float(0)
    df['mfcc13-mean'] = float(0)
    df['mfcc13-var'] = float(0)
    df['mfcc14-mean'] = float(0)
    df['mfcc14-var'] = float(0)
    df['mfcc15-mean'] = float(0)
    df['mfcc15-var'] = float(0)
    df['mfcc16-mean'] = float(0)
    df['mfcc16-var'] = float(0)
    df['mfcc17-mean'] = float(0)
    df['mfcc17-var'] = float(0)
    df['mfcc18-mean'] = float(0)
    df['mfcc18-var'] = float(0)
    df['mfcc19-mean'] = float(0)
    df['mfcc19-var'] = float(0)
    df['mfcc20-mean'] = float(0)
    df['mfcc20-var'] = float(0)

    path = f'{os.getcwd()}/wav'
 
    os.chdir(path)
   
    count = 0
    hun_Count = 0
    cursor = 1
    found = 0
    nofound = 0
    
    ## LIST OF SONG CLASSES ##
    list_Songs = []
    
    df = df.drop_duplicates(subset=['song', 'artist'])
    df['song'] = df['song'].str.replace('\(', '')
    df['song'] = df['song'].str.replace('\)', '')
    print(df.shape)
    load_Bar = f'[___________________]'

    for song in os.listdir():
        if song.endswith(".wav"):
            #print(count)
            split_Song = song.split('_')
            artist = split_Song[1].split('.')
            #print(split_Song[0] + ' ' + artist[0])
            deb = find_in_DF(df,str(split_Song[0]),str(artist[0]))
            if(deb[0] == True):
                found = found +1 
            else:
                nofound = nofound +1
            count = count +1
            hun_Count = hun_Count +1
            if(hun_Count==100):
                load_Bar = load_Bar[:cursor] + '|' + load_Bar[cursor+1:]
                cursor = cursor +1 
                hun_Count = 0

            print('LOADING ALL WAV FILES\n--------------------------------')
            print("\n"+load_Bar+f'{count}/{df.shape[0]}')
            print(f'\n\nFOUND: {found}/{len(df)}\nNOFOUND: {nofound}/{len(df)}')
            os.system('cls')
    print(f'\n\nFOUND: {found}/{count}\nNOFOUND: {nofound}/{count}')
    
    convert = 0
    hun_Count = 0
    cursor = 0
    overall_Time = perf_counter()

    for song in os.listdir():
        function_Time = perf_counter()
        asci_Art()
        quick_maths = (float(convert)/float(len(os.listdir()))*100)
        print(f'Converted {convert} : {len(os.listdir())} {quick_maths:.2f}%')
        print("\n"+load_Bar+f'{convert}/{len(os.listdir())}')
        if song.endswith(".wav"):
            split_Song = song.split('_')
            artist = split_Song[1].split('.')
            opt = find_in_DF(df,str(split_Song[0]),str(artist[0]))
            if(opt[0] == True):
                   
                #Import the song 
                y, sr = librosa.load(song)
                audio_file, _ = librosa.effects.trim(y)
                # print('\nStarting Multiprocessing for '+split_Song[0])
                
                f_get_Spectogram = functools.partial(get_Spectogram,audio_file,sr,split_Song[0],artist[0])
                f_mfccs = functools.partial(get_MFCCs, audio_file, sr)
                f_rolloff = functools.partial(rolloff, audio_file, sr)
                f_spectra_band = functools.partial(spectral_bandwidth,audio_file, sr)
                f_spec_cent = functools.partial(spectral_centroid, audio_file, sr)
                f_rms = functools.partial(rms, audio_file)
                f_harmonic_perc = functools.partial(harmonics_percept, audio_file)
                f_zero_cross = functools.partial(zero_Crossing, audio_file)
                f_mel = functools.partial(mel_Spectogram, y, sr)
                f_chroma = functools.partial(get_Chroma_stft_mean_var, audio_file, sr)

                with Pool() as pool:
                    res = pool.map(smap, [f_get_Spectogram, f_mfccs, f_rolloff, f_spectra_band, f_spec_cent, f_rms, f_harmonic_perc, f_zero_cross, f_mel,f_chroma])
                    # [0] - Nothing
                    # [1] - mfcc list
                    # [2] - rolloff Tup mean/var
                    # [3] - spectral bandwidth Tup mean/var
                    # [4] - centriod Tup mean/var
                    # [5] - RMS Tup mean/var
                    # [6] - -harm Tup mean/var, -perc Tup mean/var
                    # [7] - zero crozz - Tup mean/var
                    # [8] - mel - Tup mean/var
                    # [9] - chroma - Tup mean/var
                    #  #
                    #print(res)
                    mfcc_list = []
                    for i in range(len(res[1])):
                        mfcc_mean = res[1][i].mean()
                        mfcc_var = res[1][i].var()
                        mfcc_list.append((mfcc_mean,mfcc_var))

                    df.at[opt[1],'rolloff-mean'] = float(res[2][0])
                    df.at[opt[1],'rolloff-var'] = float(res[2][1])
                    df.at[opt[1],'bandwidth-mean'] = float(res[3][0])
                    df.at[opt[1],'bandwidth-var'] = float(res[3][1])
                    df.at[opt[1],'centroid-mean'] = float(res[4][0])
                    df.at[opt[1],'centroid-var'] = float(res[4][1])
                    df.at[opt[1],'rms-mean'] = float(res[5][0])
                    df.at[opt[1],'rms-var'] = float(res[5][1])
                    df.at[opt[1],'harmonic-mean'] = res[6][0]
                    df.at[opt[1],'harmonic-var'] = res[6][1]
                    df.at[opt[1],'percpt-mean'] = res[6][2]
                    df.at[opt[1],'percpt-var'] = res[6][3]
                    df.at[opt[1],'zeroCross-mean'] = res[7][0]
                    df.at[opt[1],'zeroCross-var'] = res[7][1]
                    df.at[opt[1],'mel-mean'] = float(res[8][0])
                    df.at[opt[1],'mel-var'] = float(res[8][1])
                    df.at[opt[1],'chroma-mean'] = float(res[9][0])
                    df.at[opt[1],'chroma-var'] = float(res[9][1])

                    for i in range(len(mfcc_list)):
                        tup = mfcc_list[i]
                        df.at[opt[1], f'mfcc{i+1}-mean'] = float(tup[0])
                        df.at[opt[1], f'mfcc{i+1}-var'] = float(tup[1])

            hun_Count = hun_Count +1
            if(hun_Count==100):
                load_Bar = load_Bar[:cursor] + '|' + load_Bar[cursor+1:]
                cursor = cursor +1 
                hun_Count = 0
            function_Time_Stop = perf_counter()
            print("Time elapsed in seconds as recorded by performance counter for song:", function_Time_Stop- function_Time)
            os.system('cls')
            convert = convert + 1

    overall_Time_Stop = perf_counter()
    print("Time elapsed in seconds as recorded by performance counter for all:", overall_Time_Stop- overall_Time)
    print('Done Converting.. Saving Dataframe')
    final_DF = df[df['bandwidth-mean'] != float(0)] 
    print(f'Final Shape: {final_DF.shape}')
    final_DF.to_csv('final_dataframe.csv')

  

            



   
   
    
    



    # get_Spectogram(audio_file, sr, split_Song[0])
    # print("MFCC")

    # #LIST OF MFCCS
    # mfcc = get_MFCCs(audio_file, sr)
    # for i in range(len(mfcc)):
    #     print(f'MFCC NUMBER {i+1} mean : {mfcc[i].mean()}')
    #     print(f'MFCC NUMBER {i+1} var : {mfcc[i].var()}')
    



if __name__ == '__main__':
    print('Loading...')
    get_MFCC_array_dataset()