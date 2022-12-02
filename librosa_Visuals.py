import librosa
import librosa.display
import matplotlib.pyplot as plt

def visualize_Chromagram_y_Spectogram(path):
    y, sr = librosa.load(path)
    audio_file, _ = librosa.effects.trim(y)
    stft = librosa.stft(audio_file)
    stft_db = librosa.amplitude_to_db(abs(stft))
    plt.figure(figsize=(16,6))
    librosa.display.specshow(stft_db, sr = sr, x_axis='time', y_axis='hz')
    plt.colorbar()

    chroma = librosa.feature.chroma_stft(audio_file, sr=sr)
    plt.figure(figsize=(16,6))
    librosa.display.specshow(chroma, sr = sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar()
    plt.title('Pitch and Chromagram')
    plt.show()


