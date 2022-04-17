import librosa
import scipy

genres = ["blues", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# librosa.feature.chroma_stft
# librosa.feature.chroma_cqt
# librosa.feature.chroma_cens

for g in genres:
    folder_path = "GTZAN/wav/" + g
