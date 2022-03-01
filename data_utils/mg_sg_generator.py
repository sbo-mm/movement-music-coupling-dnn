import os
import glob
import math

import cv2
import numpy as np
import tensorflow as tf

from data_utils.sg_preprocessor import loadaudio
from data_utils.sg_preprocessor import compute_magnitudespec

from sklearn.model_selection import train_test_split

__all__ = [
	"MotiongramSpectrogramGenerator",
	"get_dataset_ids",
	"get_dataset_for"
]

# Data Paths Globals
BASE_DATA_PATH = "/home/sbol13/sbol_data"
MG_SAVE_PATH   = BASE_DATA_PATH + "/motiongrams"
AU_SAVE_PATH   = BASE_DATA_PATH + "/dance_wav"
DS_SAVE_PATH   = BASE_DATA_PATH + "/datasets"

# Filename templates
TFE_FN_TEMPLATE = "mg_sg_pair_{0}x128_{1}_{2}"

# Random States
SPLT_RND_STATE = 1337

def strip_file(filename):
	return os.path.splitext(filename[filename.rindex('/') + 1:])[0]

def extract_music_id(aist_video_id):
	toks = aist_video_id.split('_')
	return toks[-2]

def get_dataset_ids(test_size=0.7):
	# Fetch the raw ids (filenames)
	ids_raw = [
		strip_file(fn) for fn in glob.iglob(f'{MG_SAVE_PATH}/*.npy') 
	]

	# Isolate the labels (i.e. the music id from AIST)
	labels_raw = [
		extract_music_id(id_) for id_ in ids_raw 
	]

	# Partition the labels into training/testing
	X_train, X_test, _, y_test = train_test_split(
		ids_raw, labels_raw, test_size=test_size,
		random_state=SPLT_RND_STATE, stratify=labels_raw
	)

	# Partition into validation/test
	X_test, X_validation, _, _ = train_test_split(
		X_test, y_test, test_size=0.5,
		random_state=SPLT_RND_STATE, stratify=y_test
	)

	return X_train, X_validation, X_test 

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_example(motiongram, spectrogram):
	data = {
		"motiongram"  : _bytes_feature(serialize_array(motiongram)),
		"spectrogram" : _bytes_feature(serialize_array(spectrogram))
	}
	out = tf.train.Example(features=tf.train.Features(feature=data))
	return out

def write_to_tf_records(filename: str, generator):
	record_name = f'{filename}.tfrecords'
	tfr_writer  = tf.io.TFRecordWriter(filename)
	count = 0
	for mg_sg_pair in generator():
		out = parse_single_example(*mg_sg_pair)
		tfr_writer.write(out.SerializeToString())
		count += 1
	return count, record_name

def get_dataset_for(nfft=1024, train_size=0.7, overwrite_existing=False):
	# Setup output structures
	fout_dict = {"train": None, "validation": None, "test": None}
	cmn_ncols = nfft//2+1

	# Check if dataset already exists
	if not overwrite_existing:
		tfr_files = [strip_file(fn) for fn in glob.iglob(f'{DS_SAVE_PATH}/*')]
		existing_records = [
			(os.path.splitext(r)[0], ix) for ix, r in enumerate(tfr_files)\
				if str(cmn_ncols) in r
		]
		if existing_records:
			for record, ix in existing_records:
				toks = record.split('_')
				fout_dict[toks[-2]] = (toks[-1], f'{DS_SAVE_PATH}/{tfr_files[ix]}')
			return fout_dict

	# Fetch test/train partitions
	test_size = 1. - train_size
	X_train, X_validation, X_test = get_dataset_ids(test_size=test_size)

	# Setup data generators
	generators = {
		"train": MotiongramSpectrogramGenerator(aist_ids=X_train, nfft=nfft),
		"validation": MotiongramSpectrogramGenerator(aist_ids=X_validation, nfft=nfft),
		"test": MotiongramSpectrogramGenerator(aist_ids=X_test,  nfft=nfft)
	}

	# Store as TFRecords
	for part, gen in generators.items():
		filename = TFE_FN_TEMPLATE.format(cmn_ncols, part, gen.__len__())
		filename = f'{DS_SAVE_PATH}/{filename}'
		fout, filename = write_to_tf_records(filename, gen)
		fout_dict[part] = (fout, filename)	
	return fout_dict


class MotiongramSpectrogramGenerator:
	
	# "Static" Structure to store computed stfts.
	# We make it static because the .wav files can 
	# be repeated across test/train splits
	CMPSTFTS = {}

	def __init__(self, aist_ids, nfft, ncols=128, shuffle=True):
		self.aist_ids = aist_ids
		self.ncols = ncols
		self.nrows = nfft // 2 + 1
		self.nfft = nfft
		self.shuffle = shuffle

		# Init placeholder indexes 
		self.indexes = None
		
		# Make an "init" call to `on_epoch_end`
		self.on_epoch_end()

	def __len__(self):
		return len(self.aist_ids)
		#return math.ceil(len(self.aist_ids) / self.batch_size)

	def __getitem__(self, idx):
		# Fetch the aist id to process
		aist_id_process = self.aist_ids[self.indexes[idx]]

		# Generate the required data based on the aist id
		mg, sg = self.__on_data_generation(aist_id_process)
		return mg, sg

	def __call__(self):
		for i in range(self.__len__()):
			yield self.__getitem__(i)
			if i == self.__len__() - 1:
				self.on_epoch_end()

	@property
	def __cmpstfts(self):
		return MotiongramSpectrogramGenerator.CMPSTFTS

	def __compute_magnitudespec(self, music_id):
		# Check if we have already computed a
		# spectrogram for this music id
		if music_id in self.__cmpstfts:
			return self.__cmpstfts[music_id]

		# Load the audio data
		y = loadaudio(f'{AU_SAVE_PATH}/{music_id}.wav')

		# Compute the magnitude spectrogram.
		# Normalize straight away (btw 0-1).
		magspec = compute_magnitudespec(
			y, nfft=self.nfft, with_db_normalization=True)

		# Store in local structure to avoid
		# reloading/recomputing
		self.__cmpstfts[music_id] = magspec
		return magspec

	def __on_data_generation(self, aist_id_process):
		# Fetch the music id
		music_id = extract_music_id(aist_id_process)

		# Compute magnitude spectrogram
		magspec = self.__compute_magnitudespec(music_id)

		# Load the motiongram (stored as .npy file)
		motiongram = np.load(f'{MG_SAVE_PATH}/{aist_id_process}.npy')

		# Resize the motiongram to match the spectrograms
		# dimensionality
		motiongram = np.transpose(cv2.resize(
			motiongram.T, magspec.shape,
			interpolation=cv2.INTER_AREA 
		))

		# Return a flattened view of the data.
		return motiongram.flatten(), magspec.flatten()

	def on_epoch_end(self):
		# Re-init all indexes after each epoch
		self.indexes = np.arange(len(self.aist_ids))

		# Shuffle if flag is set
		if self.shuffle:
			np.random.shuffle(self.indexes)


