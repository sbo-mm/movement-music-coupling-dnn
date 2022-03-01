import librosa
import numpy as np

__all__ = [
	"compute_magnitudespec",
	"normalize_db_0_1",
	"inverse_normalize_db_0_1",
	"MAXDURSEC",
	"SAMPLERATE",
	"HOP_LENGTH_22050_4_SEC_128C",
	"TOPDB",
	"REFDB",
	"APMIN"
]

# AUDIOLOAD & STFT PARAMETERS
MAXDURSEC  = 4
SAMPLERATE = 22050
HOP_LENGTH_22050_4_SEC_128C = 690

# DB NORMALIZATION PARAMETERS
TOPDB = 80.
REFDB = 100.
APMIN = 1e-08
 
def loadaudio(audiofile):
	y, _ = librosa.load(
		audiofile, duration=MAXDURSEC, mono=True, sr=SAMPLERATE
	)
	y = y / np.amax(np.abs(y))
	return y	

def compute_stft(y, nfft=1024):
	stft = librosa.stft(y 		   = y, 
						n_fft      = nfft, 
						hop_length = HOP_LENGTH_22050_4_SEC_128C, 
						window     = "hamming")
	return stft

def normalize_db_0_1(magspec):
	db = librosa.amplitude_to_db(
		S      = magspec,
		ref    = REFDB,
		top_db = TOPDB
	)
	db_0_1 = (db + TOPDB) / TOPDB
	return db_0_1

def inverse_normalize_db_0_1(magspec_db_0_1):
	db = (magspec_db_0_1 - 1) * TOPDB 
	magspec = librosa.db_to_amplitude(
		S_db = db,
		ref  = REFDB
	)
	return magspec

def compute_magnitudespec(y, nfft=1024, with_db_normalization=False):
	stft = compute_stft(y, nfft)
	magspec = np.abs(stft)

	if with_db_normalization:
		magspec = normalize_db_0_1(magspec)

	return magspec
