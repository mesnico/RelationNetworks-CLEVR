from PIL import Image
import argparse
import os
from tqdm import tqdm
import h5py
import pdb
import numpy as np

parser = argparse.ArgumentParser(description='Convert CLEVR dataset in a more efficient format')
parser.add_argument('clevrdir',type=str,help='Origin Clevr Directory')
args = parser.parse_args()

src_img_folder = os.path.join(args.clevrdir,'images')
dst_img_folder = os.path.join(args.clevrdir,'conv_images')

if not os.path.exists(dst_img_folder):
	os.makedirs(dst_img_folder)

for st in tqdm(os.listdir(src_img_folder)):
	cur_set = os.path.join(src_img_folder, st)

	#dst_set = os.path.join(dst_img_folder, st)
	dst_imgs_filename = os.path.join(dst_img_folder, '{}.h5'.format(st))
	h5f = h5py.File(dst_imgs_filename, 'w')
	
	for img in tqdm(os.listdir(cur_set)):
		img_path = os.path.join(cur_set, img)
		image = Image.open(img_path).convert('RGB')
		np_image = np.array(image)
		h5f.create_dataset(img, data=np_image)

	h5f.close()
