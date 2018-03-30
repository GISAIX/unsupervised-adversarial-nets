from glob import glob
import json
import nibabel as nib
import numpy as np


# image and label loading according to given path
def load_image(image_path, label_path):
    image = nib.load(image_path).get_data()[:,:,:,0]
    label = nib.load(label_path).get_data()[:,:,:,0]

    mean_num = np.mean(image)
    deviation_num = np.std(image)
    image = (image - mean_num) / (deviation_num + 1e-5)
    mean_num = np.mean(label)
    deviation_num = np.std(label)
    label = (label - mean_num) / (deviation_num + 1e-5)
    # # label mapping [0, 500, 600, 420, 550, 205, 820, 850]
    # mapping = {0: 0, 205: 5, 420: 3, 500: 1, 550: 4, 600: 2, 820: 6, 850: 7}
    # label = np.vectorize(mapping.get)(label)

    # check shape
    if image.shape != label.shape:
        print('Image and label shapes mismatch!')
        exit(1)
    return image, label


# crop single batch with the given image and label
def crop_batch(image, label, input_size, channel=1):
    image_batch = np.zeros([1, input_size, input_size, input_size, channel], dtype='float32')
    label_batch = np.zeros([1, input_size, input_size, input_size, channel], dtype='float32')
    # randomly select cube -> boundary considered
    depth, height, width = image.shape
    depth_select = np.random.randint(depth - input_size + 1)
    height_select = np.random.randint(height - input_size + 1)
    width_select = np.random.randint(width - input_size + 1)

    crop_position = np.array([depth_select, height_select, width_select])
    label_crop = label[crop_position[0]:crop_position[0] + input_size,
                       crop_position[1]:crop_position[1] + input_size,
                       crop_position[2]:crop_position[2] + input_size]

    image_crop = image[crop_position[0]:crop_position[0] + input_size,
                       crop_position[1]:crop_position[1] + input_size,
                       crop_position[2]:crop_position[2] + input_size]

    # # rotation and flipping
    # if np.random.random() > 0.333:
    #     if np.random.random() > 0.5:
    #         if rotation:
    #             rotate_angle_list = [90, 180, 270]
    #             axes_list = [(0, 1), (0, 2), (1, 2)]
    #             _angle = rotate_angle_list[np.random.randint(3)]
    #             _axes = axes_list[np.random.randint(3)]
    #             image_crop = rotate(input=image_crop, angle=_angle, axes=_axes, reshape=False, order=1)
    #             label_crop = rotate(input=label_crop, angle=_angle, axes=_axes, reshape=False, order=0)
    #     else:
    #         if flipping:
    #             _axis = np.random.randint(3)
    #             image_crop = np.flip(image_crop, axis=_axis)
    #             label_crop = np.flip(label_crop, axis=_axis)
    # NDHWC
    image_batch[0, :, :, :, 0] = image_crop
    label_batch[0, :, :, :, 0] = label_crop
    return image_batch, label_batch


# load batches including domain information
def load_train_batches(image_filelist, label_filelist, input_size, batch_size,
                       channel=1):
    # sorting?
    image_list = []
    label_list = []
    history = dict()
    image_batch_list = []
    label_batch_list = []
    for i in range(batch_size):
        select = np.random.randint(len(image_filelist))
        name = image_filelist[select]
        if name in history:
            index = history[name]
            image_batch, label_batch = crop_batch(image_list[index], label_list[index], input_size,
                                                  channel=channel)
        else:
            image, label = load_image(image_filelist[select], label_filelist[select])
            history[name] = len(image_list)
            image_list.append(image)
            label_list.append(label)
            image_batch, label_batch = crop_batch(image, label, input_size,
                                                  channel=channel)
        image_batch_list.append(image_batch)
        label_batch_list.append(label_batch)
    return np.concatenate(image_batch_list, axis=0), np.concatenate(label_batch_list, axis=0)


def load_all_images(image_filelist, label_filelist):
    # sorting?
    image_list = []
    label_list = []
    for i in range(len(image_filelist)):
        image, label = load_image(image_filelist[i], label_filelist[i])
        image_list.append(image)
        label_list.append(label)
    return image_list, label_list


def generate_filelist(image_dir, label_dir):
    image_filelist = glob(pathname='{}/*T1.img'.format(image_dir))
    label_filelist = glob(pathname='{}/*T2.img'.format(label_dir))
    image_filelist.sort()
    label_filelist.sort()
    return image_filelist, label_filelist


def write_json(dictionary, file=False, filename=None):
    if file:
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(dictionary, f, indent=4)
        else:
            print('No output file name provided.')
    return json.dumps(dictionary, indent=4)


def load_json(string, file=False):
    if file:
        with open(string, 'r') as f:
            return json.load(f)
    else:
        return json.loads(string)


if __name__ == '__main__':
    a, b = generate_filelist('../iSeg/iSeg-2017-Training/', '../iSeg/iSeg-2017-Training/')
    print(a)
    for v in a:
        print(v)
        d = nib.load(v).get_data()
        print(d.shape)
    for x in b:
        print(x)
        d = nib.load(x).get_data()
        print(d.shape)
