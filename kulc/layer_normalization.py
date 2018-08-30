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

import keras.backend as K
from keras.layers import Layer
from keras.initializers import Ones, Zeros


class LayerNormalization(Layer):
    """
        Implementation according to:
            "Layer Normalization" by JL Ba, JR Kiros, GE Hinton (2016)

    """

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        self._g = self.add_weight(
            name='gain', 
            shape=(input_shape[-1],),
            initializer=Ones(),
            trainable=True
        )
        self._b = self.add_weight(
            name='bias', 
            shape=(input_shape[-1],),
            initializer=Zeros(),
            trainable=True
        )
        
    def call(self, x):
        mean = K.mean(x, axis=-1)
        std = K.std(x, axis=-1)

        if len(x.shape) == 3:
            mean = K.permute_dimensions(
                K.repeat(mean, x.shape.as_list()[-1]),
                [0,2,1]
            )
            std = K.permute_dimensions(
                K.repeat(std, x.shape.as_list()[-1]),
                [0,2,1] 
            )
            
        elif len(x.shape) == 2:
            mean = K.reshape(
                K.repeat_elements(mean, x.shape.as_list()[-1], 0),
                (-1, x.shape.as_list()[-1])
            )
            std = K.reshape(
                K.repeat_elements(mean, x.shape.as_list()[-1], 0),
                (-1, x.shape.as_list()[-1])
            )
        
        return self._g * (x - mean) / (std + self._epsilon) + self._b