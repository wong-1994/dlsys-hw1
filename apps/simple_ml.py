"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # Read image file
    with gzip.open(image_filesname, "rb") as img_file:
        img_magic, = struct.unpack(">i", img_file.read(4))
        if img_magic != 2051:
            raise ValueError(f"MSB format parse fail.\
                 Expect 2051, but got {img_magic}")

        n_imgs, = struct.unpack(">i", img_file.read(4))

        n_rows, = struct.unpack(">i", img_file.read(4))
        n_cols, = struct.unpack(">i", img_file.read(4))
        if n_rows != 28 or n_cols != 28:
            raise ValueError(f"Data format parse fail.\
                Expect 28*28, but got {n_rows}*{n_cols}")

        X = np.empty((n_imgs, n_rows * n_cols), dtype = np.float32)
        for i in range(n_imgs):
            for j in range(n_rows * n_cols):
                X[i][j], = struct.unpack("B", img_file.read(1))
        X = X / 255.0

    # Read label file
    with gzip.open(label_filename) as label_file:
        label_magic, = struct.unpack(">i", label_file.read(4))
        if label_magic != 2049:
            raise ValueError(f"MSB format parse fail.\
                Expect 2049, but got {label_magic}")

        n_labels, = struct.unpack(">i", label_file.read(4))
        if n_labels != n_imgs:
            raise ValueError(f"Wrong number of labels.\
                Expect {n_imgs}, but got {n_labels}")

        y = np.empty(n_labels, dtype = np.uint8)
        for i in range(n_labels):
            y[i], = struct.unpack("B", label_file.read(1))
        
    return (X, y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    log_sum = ndl.log(ndl.summation(ndl.exp(Z), axes = (1,)))
    z_y = ndl.summation(ndl.multiply(Z, y_one_hot), axes = (1,))
    return ndl.summation(log_sum - z_y)/ Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_batches = (num_examples / batch) if (num_examples % batch == 0) else (num_examples / batch + 1)
    for i in range(num_batches):
        batch_start = i * batch
        batch_end = min((i + 1) * batch, num_examples)
        z_batch = ndl.matmul(ndl.relu(ndl.matmul(X[batch_start : batch_end], W1)), W2)
        y_batch = y[batch_start : batch_end]
        loss, _ = loss_err(z_batch, y_batch)
        loss.backward()
        W1.data = W1.data - lr * W1.grad.data
        W2.data = W2.data - lr * W2.grad.data
    return tuple(W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
