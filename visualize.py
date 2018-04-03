from glob import glob
import cv2
import nibabel as nib
import numpy as np


# This is only for visualization


def generate_filelist(image_dir, label_dir):
    t1_filelist = glob(pathname='{}/*T1.img'.format(image_dir))
    t2_filelist = glob(pathname='{}/*T2.img'.format(image_dir))
    label_filelist = glob(pathname='{}/*label.img'.format(label_dir))
    t1_filelist.sort()
    t2_filelist.sort()
    label_filelist.sort()
    for i, label_filename in enumerate(label_filelist):
        print(t1_filelist[i], t2_filelist[i], label_filename)
    return t1_filelist, t2_filelist, label_filelist


def extract_image(t1_path, t2_path, label_path):
    t1 = nib.load(t1_path).get_data()[:, :, :, 0]
    t2 = nib.load(t2_path).get_data()[:, :, :, 0]
    label = nib.load(label_path).get_data()[:, :, :, 0]
    print(t1.shape, t2.shape, label.shape)
    return t1, t2, label


def slice_visualize(image):
    print('Range:', np.min(image), np.max(image))
    print('Unique', np.unique(image))
    min_num = np.min(image)
    max_num = np.max(image)
    image = (image - min_num) / (min_num + max_num) * 255
    image = image.astype('uint8')
    print('Range:', np.min(image), np.max(image))
    print('Unique', np.unique(image))
    '''visualization'''
    dimension = np.asarray(image.shape)
    for k in range(dimension[2]):
        cv2.imshow('Slice', image[:, :, k])
        cv2.waitKey(0)


def visualize():
    location = '../iSeg/iSeg-2017-Training/'
    t1_filelist, t2_filelist, label_filelist = generate_filelist(location, location)
    for i in range(len(label_filelist)):
        t1, t2, label = extract_image(t1_filelist[i], t2_filelist[i], label_filelist[i])
        slice_visualize(label)
        slice_visualize(t1)
        slice_visualize(t2)
        break


if __name__ == '__main__':
    visualize()
