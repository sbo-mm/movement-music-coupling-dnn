import glob
import os

from data_utils.mg_preprocessor import transform_video2motiongram
from data_utils.sg_preprocessor import sg_test_func

import matplotlib.pyplot as plt

VIDEO_DATA_PATH = "/home/sbol13/sbol_data/aist_db_4_sec"

def main():
	vidfiles = os.listdir(VIDEO_DATA_PATH)
	vidfile = VIDEO_DATA_PATH + "/" + vidfiles[0]
	#vidarr, _ = mg_video2numpy(vidfile)
	print(len(vidfiles))
	
	fl, hl = 12, 2
	gx, gy = transform_video2motiongram(vidfile, fl, hl)
	print(gx.shape, gy.shape)

	fig, axs = plt.subplots(1, 1, figsize=(10, 10))
	axs.set_xticks([])
	axs.set_yticks([])
	axs.set_xlabel("(a) Input Motiongram")
	axs.imshow(gx, aspect="auto", cmap="binary", interpolation="bicubic")
	plt.show()


if __name__ == '__main__':
	main()