import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import librosa.display
from scipy.io import wavfile
import sunau
import numpy as np
import librosa


# def graph_spectrogram(au_file,destination):
#     frameRate, data = get_wav_info(au_file)
#     nfft = 3072  # Length of the windowing segments
#     overlap = nfft/2 # 50% overlap
#     fs = frameRate    # Sampling frequency
#     pxx, freqs, bins, im = plt.specgram(data, nfft,fs, window = mlab.window_hanning, noverlap = overlap)
#     mel_filtered = librosa.feature.melspectrogram(S = pxx, n_mels = 40, sr = frameRate, n_fft = nfft)
#     plt.plot(mel_filtered)
#     plt.axis('off')
#     plt.savefig(destination,
#                 dpi=100, # Dots per inch
#                 frameon='false',
#                 aspect='normal',
#                 bbox_inches='tight',
#                 pad_inches=0) # Spectrogram saved as a .png 


def graph_spectrogram(au_file,destination):
    y, sr = librosa.load(au_file, mono=True)
    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr = sr,
        n_mels = 40,
        n_fft = 3072)
    spectrogram = librosa.power_to_db(
        spectrogram, ref=np.max)

    plt.figure()
    librosa.display.specshow(spectrogram)
    plt.tight_layout()
    plt.savefig(destination,
            dpi=100, # Dots per inch
            frameon='false',
            aspect='normal',
            bbox_inches='tight',
            pad_inches=0) # Spectrogram saved as a .png



def get_wav_info(au_file):
    auInfo = sunau.open(au_file, 'r')
    frameRate = auInfo.getframerate()
    print("frameRate: " + str(frameRate))
    data = auInfo.readframes(auInfo.getnframes())
    audio_data = np.fromstring(data, dtype=np.dtype('>h'))
    auInfo.close()
    print(audio_data) 
    # rate, data = wavfile.read(wav_file)
    return frameRate, audio_data

def process_dir(path, prefix):
    i = 0 
    while( i < 100):
        if ( i < 10):
            numString = "0" + str(i) 
        else: 
            numString = str(i)
        fileName = prefix + ".000" + numString+".au"
        destination = "genres/"+prefix+"/spectrograms/"+prefix+numString+".png"
        graph_spectrogram(path+fileName, destination)
        i = i+1
        
    
if __name__ == '__main__': # Main function
    process_dir("genres/hiphop/", "hiphop")
    process_dir("genres/blues/", "blues")
    process_dir("genres/classical/", "classical")
    process_dir("genres/jazz/", "jazz")
    process_dir("genres/country/", "country")
    process_dir("genres/disco/", "disco")
    process_dir("genres/metal/", "metal")
    process_dir("genres/pop/", "pop")
    process_dir("genres/reggae/", "reggae")
    process_dir("genres/rock/", "rock")









