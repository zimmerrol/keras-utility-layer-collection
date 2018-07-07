# Keras Utility & Layer Collection [WIP]
Collection of custom layers for Keras which are missing in the main framework. These layers might be useful to reproduce current state-of-the-art deep learning papers using Keras.

## Applications

Using this library the following research papers have been reimplemented in Keras:

- [Attention is all you need](https://github.com/FlashTek/attention-is-all-you-need-keras)
- [Show, attend and tell](https://github.com/FlashTek/show-attend-and-tell-keras)

## Overview of implemented Layers

At the moment the `Keras Layer Collection` offers the following layers/features:

- [Scaled Dot-Product Attention](#sdpattention)
- [Multi-Head Attention](#mhatn)
- [Layer Normalization](#layernorm)
- [Sequencewise Attention](#seqatn)
- [Attention Wrapper](#atnwrapper)

### Scaled Dot-Product Attention <a name="sdpattention"></a>

Implementation as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Performs a non-linear transformation on the values `V` by comparing the queries `Q` with the keys `K`. The illustration below is taken from the paper cited above.

<img src="https://i.imgur.com/7zDGedN.jpg" height=250>

### Multi-Head Attention <a name="mhatn"></a>
Implementation as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This is basically just a bunch a [Scaled Dot-Product Attention](#sdpattention) blocks whose output is combined with a linear transformation. The illustration below is taken from the paper cited above.

<img src="https://i.imgur.com/c0xLAfS.jpg" height=250>

### Layer Normalization <a name="layernorm"></a>


### Sequencewise Attention <a name="seqatn"></a>
This layer applies various attention transformations on data. It needs a time-series of queries and a time-series of values to calculate the attention and the final linear transformation to obtain the output. This is a faster version of the general attention technique. It is similar to the `global attention` method described in [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

### Attention Wrapper <a name="atnwrapper"></a>
The idea of the implementation is based on the paper [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025). This layer can be wrapped around any `RNN` in `Keras`. It calculates for each time step of the `RNN` the attention vector between the previous output and all input steps. This way, a new attention-based input for the `RNN` is constructed. This input is finally fed into the `RNN`. This technique is similar to the `input-feeding` method described in the paper cited. The illustration below is taken from the paper cited above.

<img src="https://i.imgur.com/AZKWSd2.png" height=300>
