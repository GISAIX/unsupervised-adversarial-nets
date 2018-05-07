import numpy as np
import matplotlib.pyplot as plt


def loss_function():
    print(f'Prob: {0.005} \t Loss: {-np.log(0.005):.{4}}')
    print(f'Prob: {0.5} \t\t Loss: {-np.log(0.5):.{4}}')
    print(f'Prob: {1} \t\t Loss: {-np.log(1):.{4}}')
    prob = np.linspace(start=0, stop=1, num=300 + 1)
    entropy = - np.log(prob + 1e-5)
    dice = 2 * prob / (1 + prob * prob + (1 - prob) * (1 - prob))
    no_dice = 2 * 0 * prob / (prob * prob + (1 - prob) * (1 - prob))
    plt.plot(prob, entropy, label=r'Cross Entropy: discriminator')
    plt.plot(1 - prob, entropy, label=r'Cross Entropy: segmenter')
    plt.plot(prob, 1 - dice, label=r'Dice loss')
    # plt.plot(prob, 1 - no_dice, label=r'Dice not exists')
    plt.xlabel(r'Prob')
    plt.ylabel(r'Loss')
    plt.axis([0, 1, -0.5, 10])
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    loss_function()
