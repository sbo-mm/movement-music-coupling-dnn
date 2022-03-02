import os
import glob
import math

import cv2
import numpy as np
import tensorflow as tf

from data_utils.sg_preprocessor import loadaudio, FNCOLS
from data_utils.sg_preprocessor import compute_magnitudespec

from sklearn.model_selection import train_test_split

__all__ = [
	"MotiongramSpectrogramGenerator",
	"get_dataset_ids",
	"get_dataset_for",
	"get_dataset_small",
	"prepare_dataset_for_training",
	"extract_dlen_from_tfr",
	"TFE_FN_TEMPLATE"
]

# Data Paths Globals
BASE_DATA_PATH = "/home/sbol13/sbol_data"
MG_SAVE_PATH   = BASE_DATA_PATH + "/motiongrams"
AU_SAVE_PATH   = BASE_DATA_PATH + "/dance_wav"
DS_SAVE_PATH   = BASE_DATA_PATH + "/datasets"

# Filename templates
NROW_TOKIDX = 3
NCOL_TOKIDX = 4
TFE_FN_TEMPLATE = "mg_sg_pair_{0}x{1}_{2}_{3}"

# Random States
SPLT_RND_STATE = 1337

# Dataset Options
SHUFFLE_BUF_SIZE = 10000
PREFECTH_ADDEND  = 100 

def strip_file(filename):
	return os.path.splitext(filename[filename.rindex('/') + 1:])[0]

def extract_music_id(aist_video_id):
	toks = aist_video_id.split('_')
	return toks[-2]

def extract_dlen_from_tfr(fn_tfr):
	return int(strip_file(fn_tfr).split('_')[-1])

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

@tf.autograph.experimental.do_not_convert
def parse_tfr_example(example, cast_to_type=None):
	data = {
		'labels'      : tf.io.FixedLenFeature([], tf.string),
		'memlen'      : tf.io.FixedLenFeature([],  tf.int64),
		'motiongram'  : tf.io.FixedLenFeature([], tf.string),
		'spectrogram' : tf.io.FixedLenFeature([], tf.string)
	}

	# Extract the contents of the example at the tfrecords file
	content = tf.io.parse_single_example(example, data)
	mg_, sg_ = content['motiongram'], content['spectrogram']
	shapelen = content['memlen']
	
	mg_feature = tf.io.parse_tensor(mg_, out_type=tf.float32)
	mg_feature = tf.reshape(mg_feature, shape=[shapelen])
	sg_feature = tf.io.parse_tensor(sg_, out_type=tf.float32)
	sg_feature = tf.reshape(sg_feature, shape=[shapelen])

	# Cast to another type if desired
	if cast_to_type is not None:
		mg_feature = tf.cast(mg_feature, dtype=cast_to_type)
		sg_feature = tf.cast(sg_feature, dtype=cast_to_type)

	return mg_feature, sg_feature

def get_dataset_small(filename, dtype=None):
    # Create the dataset from the .tfrecords file
    dataset = tf.data.TFRecordDataset(filename)

    # Pass every feature through a mapping function
    dataset = dataset.map(
      lambda ex: parse_tfr_example(ex, cast_to_type=dtype)
    )
    
    return dataset

def prepare_dataset_for_evaluation(filename, cast_to_type=None):
	# Fetch the necessary batching we have to do
	# with the current memory-reduction setup
	# TODO: explain memory reduction setup
	batch_size = int(strip_file(filename).split('_')[NCOL_TOKIDX])
	ds = get_dataset_small(filename, dtype=cast_to_type)
	ds = ds.batch(batch_size)
	return ds

def prepare_dataset_for_training(filename, batch_size, cast_to_type=None):
	ds = get_dataset_small(filename, dtype=cast_to_type)
	ds = ds.shuffle(SHUFFLE_BUF_SIZE)
	ds = ds.batch(batch_size)
	ds = ds.repeat()
	ds = ds.prefetch(batch_size + PREFECTH_ADDEND)
	return ds

def parse_single_example(motiongram, spectrogram, id_string: str):
	data = {
		"labels"	  : _bytes_feature(serialize_array(id_string)),	
		"memlen"      : _int64_feature(np.prod(motiongram.shape)),
		"motiongram"  : _bytes_feature(serialize_array(motiongram)),
		"spectrogram" : _bytes_feature(serialize_array(spectrogram))
	}
	out = tf.train.Example(features=tf.train.Features(feature=data))
	return out

def write_to_tf_records(filename: str, generator):
	count = 0
	record_name = f'{filename}.tfrecords'
	tfr_writer  = tf.io.TFRecordWriter(filename)
	for mg_sg_pair in generator():
		out = parse_single_example(*mg_sg_pair)
		tfr_writer.write(out.SerializeToString())
		count += 1
	return count, filename

def fetch_if_ds_exists_for(nfft_num, fout_dict):
	# Find existing .tfrecords files
	tfr_files = [strip_file(fn) for fn in glob.iglob(f'{DS_SAVE_PATH}/*')]
	
	# Sort by the number of ffts
	existing_records = [
		(os.path.splitext(r)[0], ix) for ix, r in enumerate(tfr_files)\
			if str(nfft_num) in r
	]

	# If any exists, parse a dictionary with the
	# relevant information.
	if existing_records:
		for record, ix in existing_records:
			toks = record.split('_')
			# Populate the output structure (passed by ref)
			fout_dict[toks[-2]] = (int(toks[-1]), f'{DS_SAVE_PATH}/{tfr_files[ix]}')

		# Indicate we should terminate the calling function
		return True

	# Return nothing if we should parse a new dataset.
	return False	


def get_dataset_for(nfft=1024, overlap=512, train_size=0.7, overwrite_existing=False):
	# Setup output structures
	fout_dict = {"train": None, "validation": None, "test": None}
	cmn_ncols = FNCOLS(overlap)
	cmn_nrows = nfft//2+1

	# Check if dataset already exists
	if not overwrite_existing:
		if fetch_if_ds_exists_for(cmn_ncols, fout_dict):
			return fout_dict

	# Fetch test/train partitions
	partitions = get_dataset_ids(test_size=(1. - train_size))

	# Setup data generators
	generators = {
		k: MotiongramSpectrogramGenerator(
			aist_ids=partitions[i], nfft=nfft, overlap=overlap)\
				for i, k in enumerate(fout_dict.keys())
	}

	# Store as TFRecords
	for part, gen in generators.items():
		filename = TFE_FN_TEMPLATE.format(cmn_nrows, cmn_ncols, part, gen.agg_len)
		filename = f'{DS_SAVE_PATH}/{filename}'
		fout, filename = write_to_tf_records(filename, gen)
		fout_dict[part] = (fout, filename)	
	return fout_dict


class MotiongramSpectrogramGenerator:
	
	# "Static" Structure to store computed stfts.
	# We make it static because the .wav files can 
	# be repeated across test/train splits
	CMPSTFTS = {}

	def __init__(self, aist_ids, nfft, overlap, shuffle=True):
		self.aist_ids = aist_ids
		self.nfft = nfft
		self.overlap = overlap
		self.ncols = FNCOLS(overlap)
		self.shuffle = shuffle

		# Init indexes 
		self.indexes = np.arange(len(self.aist_ids))

		# Shuffle if flag is set
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __len__(self):
		return len(self.aist_ids)

	def __getitem__(self, idx):
		# Fetch the aist id to process
		aist_id_process = self.aist_ids[self.indexes[idx]]

		# Generate the required data based on the aist id
		mg, sg = self.__on_data_generation(aist_id_process)
		return mg, sg, aist_id_process

	def __call__(self):
		for i in range(self.__len__()):
			items = self.__getitem__(i)
			for c in range(self.ncols):
				yield items[0][:, c], items[1][:, c], items[2]

			#yield self.__getitem__(i)

	@property
	def agg_len(self):
		return self.__len__() * self.ncols

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
			y=y, nfft=self.nfft, 
			overlap=self.overlap, 
			with_db_normalization=True
		)

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

		return motiongram, magspec
		# Return a flattened view of the data.
		#return motiongram.flatten(), magspec.flatten()



