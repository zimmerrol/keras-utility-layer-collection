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
from keras.layers import Layer, Dense, TimeDistributed, Concatenate, InputSpec, Wrapper, RNN
import numpy as np

class ScaledDotProductAttention(Layer):
    """
        Implementation according to:
            "Attention is all you need" by A Vaswani, N Shazeer, N Parmar (2017)

    """

    def __init__(self, return_attention=False, **kwargs):    
        self._return_attention = return_attention
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)
        
        return input_shape[1]
    
    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Layer received an input shape {0} but expected three inputs (Q, V, K).".format(input_shape))
        else:
            if input_shape[0][0] != input_shape[1][0] or input_shape[1][0] != input_shape[2][0]:
                raise ValueError("All three inputs (Q, V, K) have to have the same batch size; received batch sizes: {0}, {1}, {2}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]))
            if input_shape[0][1] != input_shape[1][1] or input_shape[1][1] != input_shape[2][1]:
                raise ValueError("All three inputs (Q, V, K) have to have the same length; received lengths: {0}, {1}, {2}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]))
            if input_shape[0][2] != input_shape[1][2]:
                raise ValueError("Input shapes of Q {0} and V {1} do not match.".format(input_shape[0], input_shape[1]))
    
    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        
        super(ScaledDotProductAttention, self).build(input_shape)
    
    def call(self, x):
        q, k, v = x
        d_k = q.shape.as_list()[2]

        # in pure tensorflow:
        # weights = tf.matmul(x_batch, tf.transpose(y_batch, perm=[0, 2, 1]))
        # normalized_weights = tf.nn.softmax(weights/scaling)
        # output = tf.matmul(normalized_weights, x_batch)
        
        weights = K.batch_dot(q,  k, axes=[2, 2])
        normalized_weights = K.softmax(weights / np.sqrt(d_k))
        output = K.batch_dot(normalized_weights, v)
        
        if self._return_attention:
            return output, normalized_weights
        else:
            return output


class MultiHeadAttention(Layer):
    """
        Implementation according to:
            "Attention is all you need" by A Vaswani, N Shazeer, N Parmar (2017)

    """

    def __init__(self, h, d_k=None, d_v=None, d_model=None, activation=None, return_attention=False, **kwargs):    
        super(MultiHeadAttention, self).__init__(**kwargs)
        
        if (type(h) is not int or h < 2):
            raise ValueError("You have to set `h` to an int >= 2.")
        self._h = h
        
        if d_model and (type(d_model) is not int or d_model < 1):
                raise ValueError("You have to set `d_model` to an int >= 1.")
        self._d_model = d_model
        
        if d_k and int (type(d_k) is not int or d_k < 1):
            raise ValueError("You have to set `d_k` to an int >= 1.")
        self._d_k = d_k
        
        if d_v and (type(d_v) is not int or d_v < 1):
            raise ValueError("You have to set `d_v` to an int >= 1.")
        self._d_v = d_v
        
        self._activation = None
        self._return_attention = return_attention
    
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)
        
        return input_shape[1]
    
    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Layer received an input shape {0} but expected three inputs (Q, V, K).".format(input_shape))
        else:
            if input_shape[0][0] != input_shape[1][0] or input_shape[1][0] != input_shape[2][0]:
                raise ValueError("All three inputs (Q, V, K) have to have the same batch size; received batch sizes: {0}, {1}, {2}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]))
            if input_shape[0][1] != input_shape[1][1] or input_shape[1][1] != input_shape[2][1]:
                raise ValueError("All three inputs (Q, V, K) have to have the same length; received lengths: {0}, {1}, {2}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]))
            if input_shape[0][2] != input_shape[1][2]:
                raise ValueError("Input shapes of Q {0} and V {1} do not match.".format(input_shape[0], input_shape[1]))
    
    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        
        d_k = self._d_k if self._d_k else input_shape[1][-1]
        d_model = self._d_model if self._d_model else input_shape[1][-1]
        
        self._q_layers = []
        self._k_layers = []
        self._v_layers = []
        self._sdp_layer = ScaledDotProductAttention(return_attention=self._return_attention)
    
        for _ in range(self._h):
            self._q_layers.append(
                TimeDistributed(
                    Dense(d_k, activation=self._activation, use_bias=False)
                )
            )
            self._k_layers.append(
                TimeDistributed(
                    Dense(d_k, activation=self._activation, use_bias=False)
                )
            )
            self._v_layers.append(
                TimeDistributed(
                    Dense(d_k, activation=self._activation, use_bias=False)
                )
            )
        
        self._output = TimeDistributed(Dense(d_model))
        #if self._return_attention:
        #    self._output = Concatenate()
        
        super(MultiHeadAttention, self).build(input_shape)
    
    def call(self, x):
        q, k, v = x
        
        outputs = []
        attentions = []
        for i in range(self._h):
            qi = self._q_layers[i](q)
            ki = self._k_layers[i](k)
            vi = self._v_layers[i](v)
            
            if self._return_attention:
                output, attention = self._sdp_layer([qi, ki, vi])
                outputs.append(output)
                attentions.append(attention)
            else:
                output = self._sdp_layer([qi, ki, vi])
                outputs.append(output)
            
        concatenated_outputs = Concatenate()(outputs)
        output = self._output(concatenated_outputs)
        
        if self._return_attention:
            attention = Concatenate()(attentions)
       
        if self._return_attention:
            return [output, attention]
        else:
            return output        


# https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
# https://arxiv.org/pdf/1508.04025.pdf
class SequenceAttention(Layer):
    """
        Takes two inputs of the shape (batch_size, T, dim1) and (batch_size, T, dim2),
        whereby the first item is the source data and the second one the key data.
        This layer then calculates for each batch's element and each time step a softmax attention 
        vector between the key data and the source data. Finally, this attention vector is multiplied
        with the source data to obtain a weighted output. This means, that the key data is used to
        interpret the source data in a special way to create an output of the same shape as the source data.
    """
    def __init__(self, similarity, kernel_initializer="glorot_uniform", **kwargs):
        super(SequenceAttention, self).__init__(**kwargs)
        if isinstance(similarity, str):
            ALLOWED_SIMILARITIES = ["additive", "multiplicative" ]
            if similarity not in ALLOWED_SIMILARITIES:
                raise ValueError("`similarity` has to be either a callable or one of the following: {0}".format(ALLOWED_SIMILARITIES))
            else:
                self._similarity = getattr(self, "_" + similarity + "_similarity")
        elif callable(similarity):
            self._similarity = similarity
        else:
            raise ValueError("`similarity` has to be either a callable or one of the following: {0}".format(ALLOWED_SIMILARITIES))
            
        self._kernel_initializer = kernel_initializer
            
    def build(self, input_shape):
        super(SequenceAttention, self).build(input_shape)
        self._validate_input_shape(input_shape)
        
        self._weights = {}
        if self._similarity == self._additive_similarity:
            self._weights["w_a"] = self.add_weight(
                name='w_a', 
                shape=(input_shape[0][-1] + input_shape[1][-1], input_shape[0][-1]),
                initializer=self._kernel_initializer,
                trainable=True
            )
            
            self._weights["v_a"] = self.add_weight(
                name='v_a', 
                shape=(1, input_shape[0][-1]),
                initializer=self._kernel_initializer,
                trainable=True
            )
            
        elif self._similarity == self._multiplicative_similarity:
            self._weights["w_a"] = self.add_weight(
                name='w_a', 
                shape=(input_shape[1][-1], input_shape[0][-1]),
                initializer=self._kernel_initializer,
                trainable=True
            )

        self.built = True
        
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)
        
        return input_shape[0]
            
    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Layer received an input shape {0} but expected two inputs (source, query).".format(input_shape))
        else:
            if input_shape[0][0] != input_shape[1][0]:
                raise ValueError("Both two inputs (source, query) have to have the same batch size; received batch sizes: {0}, {1}".format(input_shape[0][0], input_shape[1][0]))
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("Both inputs (source, query) have to have the same length; received lengths: {0}, {1}".format(input_shape[0][0], input_shape[1][0]))
        
    def call(self, x):
        source, query = x
        
        similarity = self._similarity(source, query)
        expected_similarity_shape = [source.shape.as_list()[0], source.shape.as_list()[1], source.shape.as_list()[1]]
       
        if similarity.shape.as_list() != expected_similarity_shape:
            raise RuntimeError("The similarity function has returned a similarity with shape {0}, but expected {1}".format(similarity.shape.as_list()[:2], expected_similarity_shape))
        
        score = K.softmax(similarity)
        output = K.batch_dot(score, source, axes=[1, 1])
        
        return output
    
    def _additive_similarity(self, source, query):
        concatenation = K.concatenate([source, query], axis=2)
        nonlinearity = K.tanh(K.dot(concatenation, self._weights["w_a"]))
        
        # tile the weight vector (1, 1, dim) for each time step and each element of the batch -> (bs, T, dim)
        source_shape = K.shape(source)
        vaeff = K.tile(K.expand_dims(self._weights["v_a"], 0), [source_shape[0], source_shape[1], 1])

        similarity = K.batch_dot(K.permute_dimensions(vaeff, [0, 2, 1]), nonlinearity, axes=[1, 2])
        
        return similarity

    def _multiplicative_similarity(self, source, query):
        qp = K.dot(query, self._weights["w_a"])
        similarity = K.batch_dot(K.permute_dimensions(qp, [0, 2, 1]), source, axes=[1, 2])
        
        return similarity

class AttentionRNNWrapper(Wrapper):
    """
        The idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.

        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input
        time step's data. On this a linear transformation in the same space as the input data's space
        is performed before the data is fed into the RNN cell again.

        This technique is similar to the input-feeding method described in the paper cited
    """

    def __init__(self, layer, weight_initializer="glorot_uniform", **kwargs):
        assert isinstance(layer, RNN)
        self.layer = layer
        self.supports_masking = True
        self.weight_initializer = weight_initializer
        
        super(AttentionRNNWrapper, self).__init__(layer, **kwargs)
        
    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Layer received an input with shape {0} but expected a Tensor of rank 3.".format(input_shape[0]))

    def build(self, input_shape):
        self._validate_input_shape(input_shape)

        self.input_spec = InputSpec(shape=input_shape)
        
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
            
        input_dim = input_shape[-1]

        if self.layer.return_sequences:
            output_dim = self.layer.compute_output_shape(input_shape)[0][-1]
        else:
            output_dim = self.layer.compute_output_shape(input_shape)[-1]
      
        print(input_shape, input_dim, output_dim)

        self._W1 = self.add_weight(shape=(input_dim, input_dim), name="{}_W1".format(self.name), initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, input_dim), name="{}_W2".format(self.name), initializer=self.weight_initializer)
        self._W3 = self.add_weight(shape=(2*input_dim, input_dim), name="{}_W3".format(self.name), initializer=self.weight_initializer)
        self._b2 = self.add_weight(shape=(input_dim,), name="{}_b2".format(self.name), initializer=self.weight_initializer)
        self._b3 = self.add_weight(shape=(input_dim,), name="{}_b3".format(self.name), initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(input_dim,1), name="{}_V".format(self.name), initializer=self.weight_initializer)
        
        super(AttentionRNNWrapper, self).build()
        
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        return self.layer.compute_output_shape(input_shape)
    
    def step(self, x, states):        
        h = states[0]
        # states[1] necessary?
        
        # comes from the constants
        # equals K.dot(X, self._W1) + self._b2 with X.shape=[bs, T, input_dim]
        X = states[2]
        total_x_prod = states[3]
        
        # expand dims to add the vector which is only valid for this time step
        # to total_x_prod which is valid for all time steps
        hw = K.expand_dims(K.dot(h, self._W2), 1)
        additive_atn = total_x_prod + hw
        attention = K.softmax(K.dot(additive_atn, self._V), axis=1)
        x_weighted = K.sum(attention * X, [1])
        
        x = K.dot(K.concatenate([x, x_weighted], 1), self._W3) + self._b3
        
        h, new_states = self.layer.cell.call(x, states[:-2])
        
        return h, new_states
    
    def call(self, x, constants=None, mask=None, initial_state=None):
        # input shape: (n_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec.shape
        
        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
        else:
            initial_states = self.layer.get_initial_state(x)
            
        if not constants:
            constants = []
            
        constants += self.get_constants(x)
        
        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )
        
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output 

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def get_constants(self, x):
        # add constants to speed up calculation
        constants = [x, K.dot(x, self._W1) + self._b2]
        
        return constants

class ExternalAttentionRNNWrapper(Wrapper):
    """
        The basic idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.

        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input
        time step's data. On this a linear transformation in the same space as the input data's space
        is performed before the data is fed into the RNN cell again.

        This technique is similar to the input-feeding method described in the paper cited.

        The only difference compared to the AttentionRNNWrapper is, that this layer
        applies the attention layer not on the time-depending input but on a second
        time-independent input (like image clues) as described in:
            Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
            https://arxiv.org/abs/1502.03044
    """
    def __init__(self, layer, weight_initializer="glorot_uniform", return_attention=False, **kwargs):
        assert isinstance(layer, RNN)
        self.layer = layer
        self.supports_masking = True
        self.weight_initializer = weight_initializer
        self.return_attention = return_attention
        self._num_constants = None

        super(ExternalAttentionRNNWrapper, self).__init__(layer, **kwargs)

        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        
    def _validate_input_shape(self, input_shape):

        if len(input_shape) >= 2:
            if len(input_shape[:2]) != 2:
                raise ValueError("Layer has to receive two inputs: the temporal signal and the external signal which is constant for all time steps")
            if len(input_shape[0]) != 3:
                raise ValueError("Layer received a temporal input with shape {0} but expected a Tensor of rank 3.".format(input_shape[0]))
            if len(input_shape[1]) != 3:
                raise ValueError("Layer received a time-independent input with shape {0} but expected a Tensor of rank 3.".format(input_shape[1]))
        else:
            raise ValueError("Layer has to receive at least 2 inputs: the temporal signal and the external signal which is constant for all time steps")

    def build(self, input_shape):
        self._validate_input_shape(input_shape)

        for i, x in enumerate(input_shape):
            self.input_spec[i] = InputSpec(shape=x)
        
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
            
        temporal_input_dim = input_shape[0][-1]
        static_input_dim = input_shape[1][-1]

        if self.layer.return_sequences:
            output_dim = self.layer.compute_output_shape(input_shape[0])[0][-1]
        else:
            output_dim = self.layer.compute_output_shape(input_shape[0])[-1]
      
        self._W1 = self.add_weight(shape=(static_input_dim, temporal_input_dim), name="{}_W1".format(self.name), initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, temporal_input_dim), name="{}_W2".format(self.name), initializer=self.weight_initializer)
        self._W3 = self.add_weight(shape=(temporal_input_dim + static_input_dim, temporal_input_dim), name="{}_W3".format(self.name), initializer=self.weight_initializer)
        self._b2 = self.add_weight(shape=(temporal_input_dim,), name="{}_b2".format(self.name), initializer=self.weight_initializer)
        self._b3 = self.add_weight(shape=(temporal_input_dim,), name="{}_b3".format(self.name), initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(temporal_input_dim, 1), name="{}_V".format(self.name), initializer=self.weight_initializer)
        
        super(ExternalAttentionRNNWrapper, self).build()
        
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        output_shape =  self.layer.compute_output_shape(input_shape[0])

        if self.return_attention:
            if not isinstance(output_shape, list):
                output_shape = [output_shape]

            output_shape = output_shape + [(None, input_shape[1][1])]

        return output_shape
    
    def step(self, x, states):        
        h = states[0]
        # states[1] necessary?
        
        # comes from the constants
        X_static = states[2]
        # equals K.dot(static_x, self._W1) + self._b2 with X.shape=[bs, L, static_input_dim]
        total_x_static_prod = states[3]

        # expand dims to add the vector which is only valid for this time step
        # to total_x_prod which is valid for all time steps
        hw = K.expand_dims(K.dot(h, self._W2), 1)
        additive_atn = total_x_static_prod + hw
        attention = K.softmax(K.dot(additive_atn, self._V), axis=1)
        static_x_weighted = K.sum(attention * X_static, [1])
        
        x = K.dot(K.concatenate([x, static_x_weighted], 1), self._W3) + self._b3

        h, new_states = self.layer.cell.call(x, states[:-2])
        
        # append attention to the states to "smuggle" it out of the RNN wrapper

        attention = K.squeeze(attention, -1)

        h = K.concatenate([h, attention])

        return h, new_states
    
    def call(self, x, constants=None, mask=None, initial_state=None):
        # input shape: (n_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape

        if len(x) > 2:
            initial_state = x[2:]
            x = x[:2]
            assert len(initial_state) >= 1

        static_x = x[1]
        x = x[0]

        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
        else:
            initial_states = self.layer.get_initial_state(x)
            
        if not constants:
            constants = []
        constants += self.get_constants(static_x)
        
        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )

        # output has at the moment the form:
        # (real_output, attention)
        # split this now up

        output_dim = self.layer.compute_output_shape(input_shape)[0][-1]
        last_output = last_output[:output_dim]

        attentions = outputs[:, :, output_dim:]
        outputs = outputs[:, :, :output_dim]
        
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output 

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            output = [output] + states

        if self.return_attention:
            if not isinstance(output, list):
                output = [output]
            output = output + [attentions]

        return output

    def _standardize_args(self, inputs, initial_state, constants, num_constants):
        """Standardize `__call__` to a single list of tensor inputs.

        When running a model loaded from file, the input tensors
        `initial_state` and `constants` can be passed to `RNN.__call__` as part
        of `inputs` instead of by the dedicated keyword arguments. This method
        makes sure the arguments are separated and that `initial_state` and
        `constants` are lists of tensors (or None).

        # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None

        # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
        """
        if isinstance(inputs, list) and len(inputs) > 2:
            assert initial_state is None and constants is None
            if num_constants is not None:
                constants = inputs[-num_constants:]
                inputs = inputs[:-num_constants]
            initial_state = inputs[2:]
            inputs = inputs[:2]

        def to_list_or_none(x):
            if x is None or isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        initial_state = to_list_or_none(initial_state)
        constants = to_list_or_none(constants)

        return inputs, initial_state, constants

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an ExternalAttentionRNNWrapper'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = inputs + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(ExternalAttentionRNNWrapper, self).__call__(full_input, **kwargs)
            self.input_spec = self.input_spec[:len(original_input_spec)]
            return output
        else:
            return super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)

    def get_constants(self, x):
        # add constants to speed up calculation
        constants = [x, K.dot(x, self._W1) + self._b2]
        
        return constants

    def get_config(self):
        config = {'return_attention': self.return_attention, 'weight_initializer': self.weight_initializer}
        base_config = super(ExternalAttentionRNNWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
