from pydub import AudioSegment
import os
import wave
import glob

def check_SampleRate(file_name):
    with wave.open(file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        print(frame_rate)

def convert_Samples(file_name):
    path = f'wav/{file_name}'
    if(os.path.exists(path) == False):
        #print("Converting "+ file_name)
        sound = AudioSegment.from_file(file_name, start_second=30, duration=60) 
        sound.export(path, format="wav")
        #print("Converted "+ file_name)
    else:
        print("Already converted " + file_name)

    
    



def main():
    sample_Rate = 44100

    print(os.getcwd())
    res = [f for f in glob.glob("*.wav") if "_" in f]
    count = 0
    for songs in res:
        print(f'Converted {count} : {len(res)}')
        convert_Samples(songs)
        count = count + 1
    
    print("DONE!")

        
    # for x in os.listdir():
    #     if x.endswith(".wav"):
    #         # Prints only text file present in My Folder
    #         print(x)
            

    # src = "4 minutes (feat.wav"
    # #check_SampleRate(src)
    # dst = "wav/ult.wav"
    # # convert wav to mp3                                                            
    # sound = AudioSegment.from_file(src)
    # sound.export(dst, format="wav")
    #check_SampleRate(dst)




if __name__ == '__main__':
    main()