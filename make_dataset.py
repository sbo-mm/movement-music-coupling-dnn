import joblib
import glob
import os

import cv2
import numpy as np

#from data_utils.sg_preprocessor import loadaudio, compute_magnitudespec
#from data_utils.sg_preprocessor import normalize_db, inverse_normalize_db

# Data Paths Globals
BASE_DATA_PATH = "/home/sbol13/sbol_data/"
TMP_DATA_PATH  = "tmpdata/"
DATASETS_PATH  = "datasets/"
MOTIONGRAMS_DATA_PATH = BASE_DATA_PATH + TMP_DATA_PATH + "/motiongrams.pkl"
AUDIO_DATA_PATH       = BASE_DATA_PATH + "/dance_wav"

# Motiongram save path
MG_SAVE_PATH = BASE_DATA_PATH + "motiongrams/"

# DATASET Savenames
DSET0 = "linspec129x128.pkl"
DSET1 = "linspec513x128.pkl"

def load_audio_data():
	'''
		Load audio using librosa. Since this computation happens
		fast, we load audio lazily, i.e. when we need it.
	'''

	# Setup structure to store the audio data.
	# Use dict to match audio with correct motiongram (i.e. by key)
	audio_data_dict = {}
	
	for path in glob.iglob(AUDIO_DATA_PATH + "/*.wav"):
		# Load the audio data
		# (parameters in file `sg_preprocessor.py`)
		y = loadaudio(path)

		# Strip everything but the filename.
		# Filename is used a dict key for matching later.
		ridx = path.rindex('/') + 1
		audio_key, _ = os.path.splitext(path[ridx:])

		# Store in dict with the given key
		audio_data_dict[audio_key] = y
	return audio_data_dict 

def load_motiongram_data():
	'''
		Motiongrams has been pre-computed from their associated
		videofiles to reduce construction time. Parameters and 
		computations can be seen in `mg_preprocessor.py` and 
		`data_tester.ipynb`
	'''
	motiongram_data_dict = joblib.load(MOTIONGRAMS_DATA_PATH)
	return motiongram_data_dict


def construct_data_set(mg_data, audio_data, nfft):
	# Fetch number of eventual rows
	nrows = (nfft // 2) + 1

	# Hardcoded num of cols. 
	# See `sg_preprocessor`.
	ncols = 128 

	# Pre-allocate array(s) to store dataset
	nexamples = len(mg_data)

	# Make a structure to store the audio (stft-matrix)
	# for which we already computed an stft. 
	has_stft = {}

	# Iterate through all motiongrams and match with
	# it's associated audio track.


def main():
	# Load audio data into memory
	# print("loading audio data...")
	# audio_data_dict = load_audio_data()

	# Load motiongrams into memory
	print("loading motiongram data...")
	motiongram_data_dict = load_motiongram_data()
	print("all data loaded.")

	for mg_key, mg_val in motiongram_data_dict.items():
		mg_save_id  = "{0}.npy".format(mg_key)
		output_path = MG_SAVE_PATH + mg_save_id

		# Fetch only horz motiongrams
		mg_horz = mg_val[1]
		np.save(output_path, mg_horz)
		print("Saved {0} to {1}".format(mg_key, output_path))

if __name__ == '__main__':
	main()


