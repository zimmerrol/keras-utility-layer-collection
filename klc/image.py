# Copyright (c) 2018 Roland Zimmermann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from keras import backend as K
from keras.layers import Layer

class DiscreteResize2D(Layer):
    """
        Resizes an image by upscaling it with integer factors using a bilinear
        approximation.
    """
    
    def __init__(self, upscaling_factor, *kwargs):
        """Creates a new DiscreteResize2D layer.
        
        Arguments:
            upscaling_factor {(int, int) or int} -- Integer or pair pf integers describing
                                                    the upscaling factors of the resulting
                                                    images.
        
        Raises:
            ValueError -- ValueError will be raised if arguments are invalid.
        """


        if isinstance(upscaling_factor, int):
            self._upscaling_factors = (upscaling_factor, upscaling_factor)
        elif isinstance(upscaling_factor, (tuple, list)):
            if len(upscaling_factor) != 2:
                raise ValueError("`upscaling_factor` must either be an integer or a list/tuple of length 2.")
            self._upscaling_factors = upscaling_factor
        else:
            raise ValueError("`upscaling_factor` must either be an integer or a list/tuple of length 2.")
        super().__init__(*kwargs)

    def build(self, input_shape):
        filter_sizes = [2 * f - f % 2 for f in self._upscaling_factors]

        n_channels = input_shape[-1]

        self._W = self.add_weight(
            shape=(filter_sizes[0], filter_sizes[1], n_channels, n_channels),
            name="W",
            initializer=DiscreteResize2D._bilinear_upsampling_weights,
            trainable=False
        )

    @staticmethod
    def _bilinear_upsampling_weights(weight_shape):
        # weight_shape must be (width, height, n_channels, n_channels)

        if weight_shape[-1] != weight_shape[-2]:
            raise ValueError("Number of input channels must be the same as the number of input channels.")

        weight = np.zeros(weight_shape, dtype=np.float32)

        # create single upsampling kernel for one channel
        # according to http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/

        grid = np.ogrid[:weight_shape[0], :weight_shape[1]]
        factors = [(s+1)//2 for s in weight_shape[:2]]
        centers = [(s+1)//2 - 0.5*(s%2 + 1) for s in weight_shape[:2]]

        upsampling_kernel = (1-abs(grid[0] - centers[0]) / factors[0]) * (1-abs(grid[1] - centers[1]) / factors[1])

        for i in range(weight_shape[-1]):
            weight[:, :, i, i] = upsampling_kernel

        return weight

    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Input data has to have the dimensions(batch_size, width, height, channels).")

    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        return (self._upscaling_factors[0]*input_shape[1], self._upscaling_factors[1]*input_shape[2], input_shape[-1])

    def call(self, x):
        shape = self.compute_output_shape(x.shape.as_list())
        batch_size = K.shape(x)[0]
        output_shape = (batch_size, *shape)
      
        return K.conv2d_transpose(x, self._W, output_shape=output_shape, strides=tuple(self._upscaling_factors), padding="same")

class Resize2D(Layer):
    """ 
        Resizes an image into a new size which can is defined as a pair of integers.
        To use this layer the TensorFlow backend has to be used, as this layer only wraps
        the resize_images method of TensorFlow. The layer supports the same resizing methods
        as the TensorFlow function.
    """

    def __init__(self, output_shape, method="bilinear", *kwargs):
        """Initializes a new Resize2D layer.
        
        Arguments:
            output_shape {(int, int)} -- 2D tuple with the (height, width) of the new image
        
        Keyword Arguments:
            method {str} -- The method to use to resize the image. (default: {"bilinear"})
                            Possible values are: bilinear, nearest_neighbor, bicubic, area
        
        Raises:
            ValueError -- ValueError will be raised if arguments are invalid.
        """

        if not isinstance(output_shape, (list, tuple)) or len(output_shape) != 2:
            raise ValueError("`output_shape` must be a tuple or list of length 2.")

        if K.backend() != "tensorflow":
            raise ValueError("This layer is only supported using the tensorflow backend.")

        global tf
        import tensorflow as tf
        allowed_methods = {
            "bilinear": tf.image.ResizeMethod.BILINEAR,
            "nearest_neighbor": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            "bicubic": tf.image.ResizeMethod.BICUBIC,
            "area": tf.image.ResizeMethod.AREA
        }

        if method not in allowed_methods:
            raise ValueError("`mode´ has to be one of the values: {0}".format(allowed_methods.keys()))


        self._kwargs = kwargs
        self._output_shape = output_shape
        self._method = allowed_methods[method]
        super().__init__(*kwargs)


    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self._output_shape, input_shape[-1])

    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Input data has to have the dimensions(batch_size, width, height, channels).")

    def call(self, x):
        self._validate_input_shape(x.shape)
        output_shape = self.compute_output_shape(x.shape.as_list())

        upscaled_x = tf.image.resize_images(x, self._output_shape, method=self._method)

        return upscaled_x

    