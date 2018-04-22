# unsupervised-adversarial-nets
Function of each program:

- analysis.py: only for comprehension
- conv.py: definition for convolution, and conv-bn-relu
- inference.py: evaluation including gradient loss, l2 norm loss, and discriminator performance, the inference is conducted by dividing the original images into pieces with each of size of 32*32*32
- init.py: for some parameter definition
- iostream.py: the image is normalized with zero mean and unit variance, the batches are randomly cropped from the image with T1 and T2 image, a comparison is performed here: batch is normalized again before feeding into the network during training
- lossfunc.py: follow the paper to implement the required loss
- main.py: the entropy of the program
- model.py: a small modification is performed here that the sigmoid is replaced by ReLU with two outputs, followed by SoftMax function
- visualize: a way for you to visualize the T1, T2 and generated T2 image

read line 52-85, it will display the inference image, T1, T2 image

file name indicated in line 68, 69

T1 T2 image path indicated in line 112
