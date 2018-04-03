import numpy as np
import matplotlib.pyplot as plt

# This is only for comprehension


def loss_function():
    print(f'Prob: {0.005} \t Loss: {-np.log(0.005):.{4}}')
    print(f'Prob: {0.5} \t\t Loss: {-np.log(0.5):.{4}}')
    print(f'Prob: {1} \t\t Loss: {-np.log(1):.{4}}')
    prob = np.linspace(start=0, stop=1, num=300 + 1)
    entropy_gen = - np.log(prob + 1e-5)
    entropy_dis = - np.log(1 - prob + 1e-5)
    plt.plot(prob, entropy_gen, 'r', label=r'Generative Loss')
    plt.plot(prob, entropy_dis, 'b', label=r'Discriminator Loss')
    plt.xlabel(r'Prob = 0.5')
    plt.ylabel(r'Loss = 0.6931')
    plt.axis([0, 1, -0.5, 10])
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    loss_function()
