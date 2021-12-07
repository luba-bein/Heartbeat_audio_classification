import librosa
import numpy as np


# reading and processing an audio file
def load_audio(file_names, sound_duration=12, sample_rate=16000):
    input_length = sample_rate * sound_duration
    try:
        sound, sample_rate = librosa.load(file_names, sr=sample_rate, duration=sound_duration, res_type='kaiser_fast')
        duration = librosa.get_duration(y=sound, sr=sample_rate)
        if (round(duration) < sound_duration):
            y = librosa.util.fix_length(sound, input_length)
    except Exception as e:
        print('File reading error')
        return ''
    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40).T, axis=0)
    feature = np.array(mfccs).reshape([-1, 1])
    return feature


# audio recording recognition using loaded neural network
def sound_detection(file_path, model):
    audio = load_audio(file_path)
    audio = np.expand_dims(audio, axis=0)
    prediction = model.predict(audio)
    prediction = np.argmax(prediction)
    return prediction
