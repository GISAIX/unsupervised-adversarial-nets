from iostream import generate_filelist, load_image
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import rotate
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import time


def extract_path(directory):
    image_filelist, label_filelist = generate_filelist(directory, directory)
    for i, image_filename in enumerate(image_filelist):
        print(image_filename, label_filelist[i])
    return image_filelist, label_filelist


def extract_image(image_path, label_path):
    image, label = load_image(image_path, label_path, scale=0.6)
    print(image.shape, label.shape, image.dtype, label.dtype)
    return image, label


def slice_visualize(image):
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
    location = '../MM-WHS/mr_train/'
    image_filelist, label_filelist = extract_path(location)
    for i in range(len(image_filelist)):
        img, gt = extract_image(image_filelist[i], label_filelist[i])
        slice_visualize(gt)
        slice_visualize(img)
        # print(np.unique(gt, return_counts=True))
        break


def create_3d_view(location, mr=False, rotation_flag=False):
    _, label_filelist = extract_path(location)
    for i, label_filename in enumerate(label_filelist):
        label = nib.load(label_filename).get_data()
        mapping = {0: 0, 205: 5, 420: 3, 500: 1, 550: 4, 600: 2, 820: 6, 850: 7, 421: 3}
        label = np.vectorize(mapping.get)(label)

        name = 'ct_'
        ratio = 0.25
        if mr:
            name = 'mr_'
            ratio = 0.5
        if rotation_flag:
            name += 'r_'
            label = rotate(input=label, angle=90, axes=(0, 2), reshape=True, order=0)
            label = rotate(input=label, angle=90, axes=(1, 2), reshape=True, order=0)
            label = rotate(input=label, angle=90, axes=(0, 1), reshape=True, order=0)

        output_shape = (np.array(label.shape) * ratio).astype(dtype='int32')
        nearest_neighbor = 0
        label = resize(image=label, output_shape=output_shape, order=nearest_neighbor,
                       preserve_range=True, mode='constant')

        model(label, name + str(i))


def model(label, name):
    print(label.shape)
    # each part as a boolean array
    cube_list = []
    for i in range(8):
        cube_list.append(label == i)
    # combine all parts into single boolean array
    voxels = (label > 0)
    # set the colors of each object
    color_array = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = np.empty(voxels.shape, dtype=object)
    for i in range(1, 8):
        colors[cube_list[i]] = color_array[i]

    start_time = time.time()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor=None)
    fig.savefig(f'plot/{name}.png', dpi=1200)
    # plt.show()
    print('plot', time.time() - start_time)


if __name__ == '__main__':
    create_3d_view('../MM-WHS/ct_train/')
    create_3d_view('../MM-WHS/mr_train/', mr=True)
    create_3d_view('../MM-WHS/mr_train/', mr=True, rotation_flag=True)
    # visualize()
