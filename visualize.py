from iostream import generate_filelist, load_image
import cv2
import numpy as np


def extract_path(directory):
    image_filelist, label_filelist = generate_filelist(directory, directory)
    for i, image_filename in enumerate(image_filelist):
        print(image_filename, label_filelist[i])
    return image_filelist, label_filelist


def extract_image(image_path, label_path):
    image, label = load_image(image_path, label_path)
    print(image.shape, label.shape, image.dtype, label.dtype)
    return image, label


def slice_visualize(image):
    # [  0 205 420 500 550 600 820 850]
    print(np.min(image), np.max(image))
    print(np.unique(image))
    min_num = np.min(image)
    max_num = np.max(image)
    image = (image - min_num) / (min_num + max_num) * 255
    image = image.astype('uint8')
    print(np.min(image), np.max(image))
    print(np.unique(image))
    '''visualization'''
    dimension = np.asarray(image.shape)
    for k in range(dimension[2]):
        cv2.imshow('Slice', image[:, :, k])
        cv2.waitKey(0)


def visualize():
    location = '../iSeg/iSeg-2017-Training/'
    image_filelist, label_filelist = extract_path(location)
    for i in range(len(image_filelist)):
        img, gt = extract_image(image_filelist[i], label_filelist[i])
        slice_visualize(gt)
        slice_visualize(img)
        break


if __name__ == '__main__':
    visualize()
