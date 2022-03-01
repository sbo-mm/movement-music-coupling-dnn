import cv2
import librosa
import numpy as np
from scipy.signal import medfilt2d

__all__ = [
	"transform_video2motiongram",
	"video2numpy"
]

MG_THRESH    = 0.05
MG_MIN_MEM_B = 4

def video2numpy(filename):
	cap = cv2.VideoCapture(filename)
	fw, fh = get_video_dimensions(cap)
	framecount = get_video_framecount(cap)
	samplerate = get_video_fps(cap)

	buf = np.empty((framecount, fh, fw), np.dtype('uint8'))
	fc = 0; ret = True
	while (fc < framecount and ret):
		ret, frame = cap.read()
		if len(frame.shape) == 3 and frame.shape[-1] == 3:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		buf[fc] = frame
		fc += 1
	cap.release()
	return np.transpose(buf), (samplerate, fw, fh) 

def get_medfilt2d(kernel_size=3):
	def _medfilt2d(arr2d):
		return medfilt2d(arr2d, kernel_size)
	return _medfilt2d

def apply_over_axes_4d(arr4d, func2d):
	# Grab index iterator
	idxiter = np.ndindex(arr4d.shape[-2:])

	# Do test calc for output shape
	r1 = func2d(arr4d[:, :, 0, 0]).shape

	# Preallocate result array
	res = np.zeros(r1+arr4d.shape[-2:])

	# Iterate and apply func
	for i, j in idxiter:
		res[..., i, j] = func2d(arr4d[..., i, j])

	return res

def compute_motiongrams(video_patch, orgdims, axis, with_filter=True):
	# Reshape to original video dimensions
	newshape = (*orgdims, *video_patch.shape[1:])
	vp = np.reshape(
		video_patch, newshape=newshape, order='f'
	).astype(np.float32)

	# Compute motionframes across patches
	motion_frames = np.abs(
		np.diff(vp, axis=axis, prepend=vp[:, :, 0:1, :])
	)

	# Apply FILTERING here
	motion_frames = (motion_frames > MG_THRESH*255)*motion_frames

	if with_filter:
		motion_frames = apply_over_axes_4d(motion_frames, get_medfilt2d())

	# Horz mg is axis 0: Collapse axis 1
	mx   = np.mean(motion_frames, axis=1)
	mx_c = np.sum(mx, axis=axis)

	# Vert mg is axis 1: Collapse axis 0
	my   = np.mean(motion_frames, axis=0)
	my_c = np.sum(my, axis=axis)

	return mx_c, my_c

def transform_video2motiongram(videoframes, frame_length, hop_length, with_filter=True):
	# Acquire videoframes in numpy format
	vframes = videoframes[0] #video2numpy(videofile)
	samplerate, width, height = videoframes[1]

	# Save original videoframe dimensions
	orgdims = (width, height)

	# Reshape to "two dimensions"
	newshape = (np.prod(vframes.shape[:-1]), -1)
	vframes = np.reshape(vframes, newshape=newshape, order='f')

	# Setup padding options
	padding = [(0, 0) for _ in range(vframes.ndim)]
	padlr = int((frame_length + 15) // 2)
	padding[-1] = (padlr, padlr)
	vframes = np.pad(vframes, padding, mode='reflect')

	# Slice the video frames into patches
	vsliced = librosa.util.frame(vframes, frame_length, hop_length)
	
	# Compute an overall shape for the output structures
	shape = [*orgdims, *vsliced.shape[-2:]]
	
	shapex = (shape[0], shape[-1])
	gramx = np.empty(shapex, dtype=np.float32, order='f')

	shapey = (shape[1], shape[-1])
	gramy = np.empty(shapey, dtype=np.float32, order='f')
	#print(gramx.shape, gramy.shape)

	# Compute how many blocks we process per iteration
	n_columns = librosa.util.MAX_MEM_BLOCK // (
		np.prod(shape[:-1]) * vsliced.itemsize
	); n_columns = max(n_columns, MG_MIN_MEM_B)

	# Process the motiongram on a patch basis
	for bl_s in range(0, shape[-1], n_columns):
		bl_t = min(bl_s + n_columns, shape[-1])
		gramx[..., bl_s:bl_t], gramy[..., bl_s:bl_t] \
			= compute_motiongrams(vsliced[..., bl_s:bl_t], orgdims, -2, with_filter=with_filter)

	# Perform postprocessing
	gramx = ((gramx-gramx.min())/(gramx.max()-gramx.min()))*255
	gramy = ((gramy-gramy.min())/(gramy.max()-gramy.min()))*255
	gramx = cv2.cvtColor(gramx.astype(
		np.uint8), cv2.COLOR_GRAY2BGR)
	gramy = cv2.cvtColor(gramy.astype(
		np.uint8), cv2.COLOR_GRAY2BGR)

	# Always equalize the motiongrams
	gramx_hsv = cv2.cvtColor(gramx, cv2.COLOR_BGR2HSV)
	gramx_hsv[:, :, 2] = cv2.equalizeHist(gramx_hsv[:, :, 2])
	gramx = cv2.cvtColor(gramx_hsv, cv2.COLOR_HSV2RGB)
	gramy_hsv = cv2.cvtColor(gramy, cv2.COLOR_BGR2HSV)
	gramy_hsv[:, :, 2] = cv2.equalizeHist(gramy_hsv[:, :, 2])
	gramy = cv2.cvtColor(gramy_hsv, cv2.COLOR_HSV2RGB)

	# Convert to single channel and cast to float32 for
	# normalization btw 0-1
	gramx = cv2.cvtColor(gramx, cv2.COLOR_RGB2GRAY).astype(
		np.float32)/255.
	gramy = cv2.cvtColor(gramy, cv2.COLOR_RGB2GRAY).astype(
		np.float32)/255.
	return gramx, gramy


############################################################
#                                                          #
#                                                          #
############################################################

def get_video_dimensions(capture_object):
	framewidth  = int(capture_object.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameheight = int(capture_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
	return framewidth, frameheight

def get_video_framecount(capture_object):
	return int(capture_object.get(cv2.CAP_PROP_FRAME_COUNT))

def get_video_fps(capture_object):
	return capture_object.get(cv2.CAP_PROP_FPS)