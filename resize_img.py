# resize imaes in a folder
import glob
import cv2
import os

 
 def resize_one(img_path, out_path):
	img = cv2.imread(img)
	print("original shape: " + img.shape) # (1280 x 960)
	cropped = img[0:960, :]  # [y0:y1, x0:x1]
	print("resize shape:" + img.shape)
	cv2.imwrite(out_path, cropped)


def resize_folder(folder_path):
	base_dir = './resized_images'
	os.mkdir(base_dir)

    img_paths = glob.glob(folder_path + '/*.jpg')
    s = len(folder_path)
    for img_path in img_paths:
    	out_path = base_dir + img_path[s:-4] + 'Re.jpg'

        resize_folder(img_path, out_path)

def main():
	folder_path = ''
	img_path = ''
	out_path = ''
	resize_one(img_path, out_path)


if __name__ == '__main__':
	main()