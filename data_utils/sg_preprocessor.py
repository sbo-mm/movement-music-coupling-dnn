import librosa
import numpy as np
from scipy.signal.windows import tukey

__all__ = [
	"loadaudio",
	"compute_magnitudespec",
	"normalize_db_0_1",
	"inverse_normalize_db_0_1",
	"spectrogram2audio",
	"FNCOLS",
	"MAXDURSEC",
	"MAXOFFSET",
	"SAMPLERATE",
	"TOPDB",
	"REFDB"
]

# AUDIOLOAD & STFT PARAMETERS
MAXDURSEC  = 4
MAXOFFSET  = 16
SAMPLERATE = 22050

# DB NORMALIZATION PARAMETERS
TOPDB = 80.
REFDB = 100.

# WINDOWS
TALPH = 0.3
TSAMP = MAXDURSEC * SAMPLERATE
TUKEY = tukey(TSAMP, alpha=TALPH)

# Convenience Lambda(s)
FNCOLS = lambda hl: int(np.ceil(MAXDURSEC * SAMPLERATE / hl))
 
def loadaudio(audiofile, offset=0):
	y, _ = librosa.load(
		audiofile, 
		offset=offset, 
		duration=MAXDURSEC, 
		mono=True, 
		sr=SAMPLERATE
	)
	y = y / np.amax(np.abs(y))
	y = np.multiply(y, TUKEY)
	return y	

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

def compute_stft(y, nfft=1024, overlap=512):
	stft = librosa.stft(
		y 		   = y, 
		n_fft      = nfft, 
		hop_length = overlap, 
		window     = "hamming"
	)
	return stft

def compute_magnitudespec(y, nfft=1024, overlap=512, with_db_normalization=False):
	stft = compute_stft(y, nfft, overlap)
	magspec = np.abs(stft)

	if with_db_normalization:
		magspec = normalize_db_0_1(magspec)

	return magspec

def spectrogram2audio(magspec, hop_length, with_inverse_normalize=False):

	if with_inverse_normalize:
		magspec = inverse_normalize_db_0_1(magspec)

	y = librosa.griffinlim(
		S          = magspec, 
		hop_length = hop_length,
		window     = "hamming"
	)
	y = y / np.amax(np.abs(y))
	return y


