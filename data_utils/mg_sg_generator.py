import os
import glob
import math

import cv2
import numpy as np
import tensorflow as tf
from librosa.util import MAX_MEM_BLOCK as libMEM_BLOCK

from data_utils import DATASET_DTYPE

from data_utils.sg_preprocessor import loadaudio, FNCOLS
from data_utils.sg_preprocessor import MAXDURSEC, MAXOFFSET
from data_utils.sg_preprocessor import compute_magnitudespec

from sklearn.model_selection import train_test_split

__all__ = [
	"MotiongramSpectrogramGenerator",
	"get_dataset_ids",
	"get_dataset_for",
	"get_dataset_small",
	"prepare_dataset_for_training",
	"prepare_dataset_for_evaluation",
	"splice_spectrogram_patches",
	"num_examples",
	"extract_dlen_from_tfr",
	"strip_file",
	"TFE_FN_TEMPLATE"
]

# Data Paths Globals
BASE_DATA_PATH = "/home/sbol13/sbol_data"
MG_SAVE_PATH   = BASE_DATA_PATH + "/motiongrams"
AU_SAVE_PATH   = BASE_DATA_PATH + "/dance_wav"
DS_SAVE_PATH   = BASE_DATA_PATH + "/datasets"

# TEMPLATE STRUCTURE:
# mg_sg_pair_ROWSxCOLS_PARTITION_ACTUALEXAMPLES_AGGREGATEDEXAMPLES_BATCHSIZE`
# e.g. mg_sg_pair_129x690_train_1054_6324_127
TFE_FN_TEMPLATE = "mg_sg_pair_{0}x{1}_{2}_{3}_{4}_{5}"

# Template indexers
ROW_COL_IDX      = -5
PARTITION_IDX    = -4
NUM_EXAMPLES_IDX = -3
AGG_EXAMPLES_IDX = -2
BATCH_SIZE_IDX   = -1

# Random States
SPLT_RND_STATE = 1337
np.random.seed(SPLT_RND_STATE)

# Dataset Options
SHUFFLE_BUF_SIZE = 10000
PREFECTH_ADDEND  = 100 

# MEMORY PARAMS
FLOAT64BYTES = 8
MAX_MEM_BLOCK = libMEM_BLOCK // 2

def num_examples():
	return len(os.listdir(f'{MG_SAVE_PATH}/'))

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
		'spectrogram' : tf.io.FixedLenFeature([], tf.string),
		'mg_dtype'    : tf.io.FixedLenFeature([], tf.string),
		'sg_dtype'    : tf.io.FixedLenFeature([], tf.string),
	}

	# Extract the contents of the example at the tfrecords file
	content = tf.io.parse_single_example(example, data)
	mg_, sg_ = content['motiongram'], content['spectrogram']
	shapelen = content['memlen']
	
	mg_feature = tf.io.parse_tensor(mg_, out_type=tf.dtypes.as_dtype(DATASET_DTYPE))
	mg_feature = tf.reshape(mg_feature, shape=[shapelen])
	sg_feature = tf.io.parse_tensor(sg_, out_type=tf.dtypes.as_dtype(DATASET_DTYPE))
	sg_feature = tf.reshape(sg_feature, shape=[shapelen])

	# Cast to another type if desired
	if (cast_to_type is not None) and (cast_to_type is not DATASET_DTYPE):
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

def prepare_dataset_for_evaluation(filename, batch_size, cast_to_type=None):
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
		"spectrogram" : _bytes_feature(serialize_array(spectrogram)),
		"mg_dtype"    : _bytes_feature(serialize_array(motiongram.dtype.name)),
		"sg_dtype"    : _bytes_feature(serialize_array(motiongram.dtype.name))
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
			fout_dict[toks[-4]] = (
				int(toks[AGG_EXAMPLES_IDX]),
				int(toks[NUM_EXAMPLES_IDX]),
				int(toks[BATCH_SIZE_IDX]),
				f'{DS_SAVE_PATH}/{tfr_files[ix]}'
			)

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
		# Parse a filename
		stmp = TFE_FN_TEMPLATE.format(
			cmn_nrows, cmn_ncols, part, gen.__len__(), gen.agg_len, gen.batch)
		filename = f'{DS_SAVE_PATH}/{stmp}'
		
		# Pass the generator for iteratively writing
		# this partition to a tf record.
		fout, filename = write_to_tf_records(filename, gen)
		
		# Return a structure to describe basics 
		# of the dataset.
		fout_dict[part] = (fout, gen.__len__(), gen.batch, filename)	
	return fout_dict

def splice_spectrogram_patches(patches, orig_cols):
	# Compute the padding applied (if any)
	# during data generation.
	new_cols = patches[0].shape[-1] * len(patches)
	padlen   = new_cols - orig_cols
	
	# Compute how much to trim from both
	# ends of the spliced array.
	padl = math.floor(padlen / 2)
	padr = math.ceil( padlen / 2)

	# Concatenate patches.
	spliced = np.concatenate(patches, axis=-1)

	# Trim padding from the last axis
	spliced = spliced[..., padl:-padr]
	return spliced

class MotiongramSpectrogramGenerator:
	
	# "Static" Structure to store computed stfts.
	# We make it static because the .wav files can 
	# be repeated across test/train splits.
	CMPSTFTS = {}

	def __init__(self, aist_ids, nfft, overlap, shuffle=True):
		self.aist_ids = aist_ids
		self.nfft = nfft
		self.overlap = overlap
		self.nrows = nfft//2+1
		self.ncols = FNCOLS(overlap)
		self.ndims = 2
		self.shuffle = shuffle

		# Setup batching options
		self.batch = MAX_MEM_BLOCK // (
			self.nrows * FLOAT64BYTES
		)

		self.padlen = 0
		self.padopt = self.ncols % self.batch
		self.padding = [(0, 0) for _ in range(self.ndims)]

		if self.padopt != 0:
			self.padlen = self.batch - self.padopt
			self.padl = math.floor(self.padlen / 2)
			self.padr = math.ceil( self.padlen / 2)
			self.padding[-1] = (self.padl, self.padr)

		# Number of batches per specgram
		self.numbatches = (self.ncols + self.padlen) // self.batch

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
			mg_batched = self.__batch_example(items[0])
			sg_batched = self.__batch_example(items[1])
			for j in range(self.numbatches):
				yield mg_batched[j], sg_batched[j], items[-1]

	@property
	def agg_len(self):
		return self.__len__() * self.numbatches

	@property
	def __cmpstfts(self):
		return MotiongramSpectrogramGenerator.CMPSTFTS

	def __batch_example(self, example):
		example_padded = np.pad(example, self.padding, mode="reflect")
		example_batched = [
			example_padded[:, i*self.batch:(i+1)*self.batch]\
				for i in range(self.numbatches)
		]
		return example_batched

	def __compute_magnitudespec(self, music_id):
		# Check if we have already computed a
		# spectrogram for this music id
		if music_id in self.__cmpstfts:
			return self.__cmpstfts[music_id]

		# Load the audio data
		y = loadaudio(
			f'{AU_SAVE_PATH}/{music_id}.wav'
		 )

		# Compute the magnitude spectrogram.
		# Normalize straight away (btw 0-1).
		magspec = compute_magnitudespec(
			y=y, nfft=self.nfft, 
			overlap=self.overlap, 
			with_db_normalization=True
		)

		# Check if we should cast the data.
		if magspec.dtype.name != DATASET_DTYPE:
			magspec = magspec.astype(DATASET_DTYPE)

		# Store in local structure to avoid
		# reloading/recomputing
		self.__cmpstfts[music_id] = magspec
		return magspec

	def __compute_motiongram(self, motiongram_id, mg_newshape):
		# Load the motiongram (stored as .npy file)
		# TODO: inline compute the motiongram instead of loading
		# precomputed.
		motiongram = np.load(f'{MG_SAVE_PATH}/{motiongram_id}.npy')

		# Resize the motiongram to match the spectrograms
		# dimensionality
		motiongram = np.transpose(cv2.resize(
			motiongram.T, mg_newshape,
			interpolation=cv2.INTER_AREA 
		))
		
		# Check if we should cast the data.
		if motiongram.dtype.name != DATASET_DTYPE:
			motiongram = motiongram.astype(DATASET_DTYPE)

		return motiongram	

	def __on_data_generation(self, aist_id_process):
		# Fetch the music id
		music_id   = extract_music_id(aist_id_process)
		magspec    = self.__compute_magnitudespec(music_id)
		motiongram = self.__compute_motiongram(aist_id_process, magspec.shape)
		return motiongram, magspec



