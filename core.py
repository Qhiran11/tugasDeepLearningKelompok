import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Layer:
    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))

    def forward(self, input):
        self.input = input
        N, C, H, W = input.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)

        self.x_col = im2col_indices(
            input, self.kernel_size, self.kernel_size, self.padding, self.stride
        )
        w_row = self.weights.reshape(self.out_channels, -1)

        out = np.dot(w_row, self.x_col) + self.bias
        out = out.reshape(self.out_channels, out_h, out_w, N)
        return out.transpose(3, 0, 1, 2)

    def backward(self, output_gradient, learning_rate):
        N, C, H, W = self.input.shape
        dout_reshaped = output_gradient.transpose(1, 2, 3, 0).reshape(
            self.out_channels, -1
        )

        dw = np.dot(dout_reshaped, self.x_col.T).reshape(self.weights.shape)

        db = np.sum(dout_reshaped, axis=1, keepdims=True)

        w_row = self.weights.reshape(self.out_channels, -1)
        dx_col = np.dot(w_row.T, dout_reshaped)
        dx = col2im_indices(
            dx_col,
            self.input.shape,
            self.kernel_size,
            self.kernel_size,
            self.padding,
            self.stride,
        )

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dx


class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        N, C, H, W = input.shape

        self.x_reshaped = input.reshape(
            N,
            C,
            H // self.pool_size,
            self.pool_size,
            W // self.pool_size,
            self.pool_size,
        )
        out = self.x_reshaped.max(axis=(3, 5))
        return out

    def backward(self, output_gradient, learning_rate=None):
        N, C, H, W = self.input.shape
        out_h = H // self.pool_size
        out_w = W // self.pool_size

        dout_reshaped = output_gradient[:, :, :, np.newaxis, :, np.newaxis]

        mask = self.x_reshaped == self.x_reshaped.max(axis=(3, 5), keepdims=True)

        d_input = mask * dout_reshaped
        return d_input.reshape(N, C, H, W)


class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape

        return input.reshape(input.shape[0], -1)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)


class Dense(Layer):
    def __init__(self, input_size, output_size):

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient) / self.input.shape[0]
        bias_gradient = (
            np.sum(output_gradient, axis=0, keepdims=True) / self.input.shape[0]
        )
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)


class Sigmoid(Layer):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-np.clip(input, -50, 50)))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.output * (1 - self.output)


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
