from iostream import load_image
import numpy as np
import time


class Evaluation:
    def __init__(self):
        self.performance = None
        self.dis_accuracy = None
        # storage
        self.error = []
        self.gradient = []
        self.accuracy = [[], []]

    def add(self, error, gradient):
        self.error.append(error)
        self.gradient.append(gradient)

    def add_domain(self, domain_accuracy):
        self.accuracy[0].append(domain_accuracy[0])
        self.accuracy[1].append(domain_accuracy[1])

    def retrieve(self):
        # shape: (2, 1)
        self.performance = np.array([self.error, self.gradient], dtype='float32')
        return np.mean(self.performance, axis=1)

    def retrieve_domain(self):
        # shape: (2, 1)
        self.dis_accuracy = np.array(self.accuracy, dtype='float32')
        return np.mean(self.dis_accuracy, axis=1)


# a passing network
def network(_, label, domain):
    # input is a numpy array
    return label.copy(), np.concatenate([domain.copy(), 1 - domain.copy()], axis=0)


def compute_gradient(images):
    # input shape: NDHWC
    pixel_diff_d = images[:, 1:, :, :, :] - images[:, :-1, :, :, :]
    pixel_diff_h = images[:, :, 1:, :, :] - images[:, :, :-1, :, :]
    pixel_diff_w = images[:, :, :, 1:, :] - images[:, :, :, :-1, :]
    gradient_d = np.abs(pixel_diff_d)
    gradient_h = np.abs(pixel_diff_h)
    gradient_w = np.abs(pixel_diff_w)
    return gradient_d, gradient_h, gradient_w


def compute_performance(inference, label):
    error = np.mean((inference - label) * (inference - label))
    gradient_gen_x, gradient_gen_y, gradient_gen_z = compute_gradient(inference)
    gradient_label_x, gradient_label_y, gradient_label_z = compute_gradient(label)
    gradient = np.mean((gradient_gen_x - gradient_label_x) * (gradient_gen_x - gradient_label_x)) + np.mean(
        (gradient_gen_y - gradient_label_y) * (gradient_gen_y - gradient_label_y)) + np.mean(
        (gradient_gen_z - gradient_label_z) * (gradient_gen_z - gradient_label_z))
    print(f'Error: {error}')
    print(f'Gradient difference: {gradient}')
    return error, gradient


def compute_domain_performance(discrimination, domain_label):
    accuracy = []
    for i in range(2):
        correct = 1 * (discrimination[i, :] == domain_label + i)
        i_accuracy = np.sum(correct) / (discrimination.shape[1] + 1e-5)
        accuracy.append(i_accuracy)
        print(f'Domain accuracy: {i_accuracy}')
    return accuracy


def infer(image, label, domain, input_size=32, strike=32, channel=1,
          infer_task=None, coefficient=None, loss_log=None, evaluation=None, sample=None):

    # # fast forwarding: 32, center-cropping: 16
    # strike, equivalent to effective

    # skip
    strike_skip = (input_size - strike) // 2
    depth, height, width = image.shape

    # axis expansion
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=4)
    label = np.expand_dims(label, axis=0)
    label = np.expand_dims(label, axis=4)
    # initialization with -1
    discrimination = [[], []]
    inference = -1 * np.ones(label.shape, label.dtype)
    image_batch = np.zeros([1, input_size, input_size, input_size, channel], dtype='float32')
    label_batch = np.zeros([1, input_size, input_size, input_size, channel], dtype='float32')
    # degenerate domain batch
    domain_batch = np.array([domain, 1 - domain], dtype=np.int32)

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
                label_batch[0, :, :, :, 0] = label[0, fetch_d:fetch_d + input_size, fetch_h:fetch_h + input_size,
                                                   fetch_w:fetch_w + input_size, 0]

                # main body of network
                if infer_task is None:
                    infer_batch, infer_domain = network(image_batch, label_batch, domain_batch)
                else:
                    infer_batch, infer_domain = infer_task(
                        image_batch, label_batch, domain_batch, coefficient, loss_log,
                        fetch_d / depth, fetch_h / height, fetch_w / width, sample=sample)

                # fast forwarding
                inference[0, put_d:put_d + size_d, put_h:put_h + size_h, put_w:put_w + size_w, 0] = \
                    infer_batch[0, -size_d:, -size_h:, -size_w:, 0]
                discrimination[0].append(infer_domain[0])
                discrimination[1].append(infer_domain[1])

    discrimination = np.array(discrimination, dtype='int32')

    print('Running time:', time.time() - start_time)

    accuracy = compute_domain_performance(discrimination, domain)
    error, gradient = compute_performance(inference, label)
    evaluation.add(error, gradient)
    evaluation.add_domain(accuracy)

    return inference


if __name__ == '__main__':
    img, truth = load_image('../iSeg/iSeg-2017-Training/subject-1-T1.img',
                            '../iSeg/iSeg-2017-Training/subject-1-T2.img')
    print('image:', img.shape)
    print('label:', truth.shape)
    e = Evaluation()
    infer(img, truth, 0, strike=32, evaluation=e)
    print(e.retrieve().shape)
    print(e.retrieve_domain().shape)
    print(str(e.retrieve()))
    print(str(e.retrieve_domain()))
