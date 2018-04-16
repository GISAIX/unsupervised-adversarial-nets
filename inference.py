from iostream import load_image
import numpy as np
import time


# a passing network
def network(_, label):
    # input is a numpy array
    inference = label.copy()
    return inference


def compute_performance(inference, label, class_num):
    dice = []
    jaccard = []
    accuracy = []
    start_time = time.time()
    for i in range(class_num):
        i_inference = 1 * (inference[:, :, :, :] == i)
        i_label = 1 * (label[:, :, :, :] == i)
        # dice
        intersection = np.sum(i_inference * i_label)
        summation = np.sum(i_inference) + np.sum(i_label) + 1e-5
        i_dice = 2 * intersection / summation
        # jaccard
        addition = i_inference + i_label
        union = np.sum(1 * (addition[:, :, :, :] > 0)) + 1e-5
        i_jaccard = intersection / union
        # accuracy
        i_accuracy = np.sum(intersection) / (np.sum(i_label) + 1e-5)

        print(f'{i}: {i_dice} {i_jaccard} {i_accuracy}')
        dice.append(i_dice)
        jaccard.append(i_jaccard)
        accuracy.append(i_accuracy)
    # time reduction needed
    print(f'time: {time.time() - start_time}')
    return dice, jaccard, accuracy


def test(image, label, strike=32):
    input_size = 32
    channel = 1

    # # fast forwarding: 32, center-cropping: 16
    # strike = input_size // 2  # if it is 16 now, equivalent to effective

    # skip
    strike_skip = (input_size - strike) // 2
    depth, height, width = image.shape

    # axis expansion
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=4)
    label = np.expand_dims(label, axis=0)
    # initialization with -1
    inference = -1 * np.ones(label.shape, label.dtype)
    image_batch = np.zeros([1, input_size, input_size, input_size, channel], dtype='float32')
    label_batch = np.zeros([1, input_size, input_size, input_size], dtype='int32')

    # -1 symbol for last
    depth_range = np.append(np.arange(depth - input_size + 1, step=strike), -1)
    height_range = np.append(np.arange(height - input_size + 1, step=strike), -1)
    width_range = np.append(np.arange(width - input_size + 1, step=strike), -1)

    start_time = time.time()
    for d in depth_range:
        for h in height_range:
            for w in width_range:
                # default situation
                fetch_d, fetch_h, fetch_w = d, h, w  # fetch variable for the last batch
                put_d, put_h, put_w = d + strike_skip, h + strike_skip, w + strike_skip  # put for inference position
                size_d, size_h, size_w = input_size - strike_skip, input_size - strike_skip, input_size - strike_skip
                # for inference length

                if d == -1:
                    if depth % strike == 0:
                        continue
                    else:
                        fetch_d = depth - input_size
                        size_d = depth % strike
                        put_d = depth - size_d
                elif d == 0:
                    put_d = d
                    size_d = input_size

                if h == -1:
                    if height % strike == 0:
                        continue
                    else:
                        fetch_h = height - input_size
                        size_h = height % strike
                        put_h = height - size_h
                elif h == 0:
                    put_h = h
                    size_h = input_size

                if w == -1:
                    if width % strike == 0:
                        continue
                    else:
                        fetch_w = width - input_size
                        size_w = width % strike
                        put_w = width - size_w
                elif w == 0:
                    put_w = w
                    size_w = input_size

                # batch cropping
                image_batch[0, :, :, :, 0] = image[0, fetch_d:fetch_d + input_size, fetch_h:fetch_h + input_size,
                                                   fetch_w:fetch_w + input_size, 0]
                label_batch[0, :, :, :] = label[0, fetch_d:fetch_d + input_size, fetch_h:fetch_h + input_size,
                                                fetch_w:fetch_w + input_size]
                # main body of network
                infer_batch = network(image_batch, label_batch)
                # fast forwarding
                inference[0, put_d:put_d + size_d, put_h:put_h + size_h, put_w:put_w + size_w] = \
                    infer_batch[0, -size_d:, -size_h:, -size_w:]

    print('time:', time.time() - start_time)
    print('image:', image.shape)
    print('label:', label.shape)
    print('inference:', inference.shape)
    print(f'strike: {strike}, equal: {np.array_equal(inference, label)}', )
    _ = compute_performance(inference, label, 8)


if __name__ == '__main__':
    img, truth = load_image('../MM-WHS/ct_train/ct_train_1001_image.nii.gz',
                            '../MM-WHS/ct_train/ct_train_1001_label.nii.gz')
    print('image:', img.shape)
    print('label:', truth.shape)
    test(img, truth, strike=32)
