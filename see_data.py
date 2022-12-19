import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display

from config import Config

def raw():
	signal, sr = librosa.load(path, sr=Config.SAMPLE_RATE)
	idx = np.where(np.abs(signal) >= Config.SER_NORM_CUT_THRESHOLD)[0][0]
	signal = signal[idx:]
	plt.plot(signal)
	plt.show()

def get_mel(signal, sample_rate):
	mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate,
	                                          n_fft=1024,
	                                          win_length=512,
	                                          window='hamming',
	                                          hop_length = 256,
                                              n_mels=128,
                                              fmax=sample_rate/2)
	mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
	return mel_spec_db



if __name__ == '__main__':
	path = './SER/normal/2018082875560637345.wav'
	# path = './SER/abnormal/Repair/Number2.wav'
	# path = './SER/abnormal/Repair/Number10.wav'

	signal, sr = librosa.load(path, sr=Config.SAMPLE_RATE, duration=15, offset=0.5)
	mel = get_mel(signal, Config.SAMPLE_RATE)
	# librosa.display.specshow(mel, y_axis='mel', x_axis='time')
	# plt.show()
	print(mel.shape)





