import numpy as np
import matplotlib.pyplot as plt


def loss_function():
    print(f'Prob: {0.005} \t Loss: {-np.log(0.005)}')
    print(f'Prob: {1} \t\t Loss: {-np.log(1)}')
    prob = np.linspace(start=0, stop=1, num=300 + 1)
    entropy = - np.log(prob + 1e-5)
    dice = 2 * prob / (1 + prob * prob + (1 - prob) * (1 - prob))
    no_dice = 2 * 0 * prob / (prob * prob + (1 - prob) * (1 - prob))
    plt.plot(prob, entropy, 'r', label=r'Cross Entropy')
    plt.plot(prob, 1 - dice, label=r'Dice exists')
    plt.plot(prob, 1 - no_dice, label=r'Dice not exists')
    plt.xlabel(r'Prob')
    plt.ylabel(r'Loss')
    plt.axis([0, 1, -0.5, 10])
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    loss_function()
