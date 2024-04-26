# transformers_nitty_gritty

Transformers are a type of deep learning model architecture primarily used for natural language processing (NLP) tasks, although they have also been applied to other domains such as computer vision. Introduced by Vaswani et al. in the paper "Attention is All You Need" in 2017, transformers have become the cornerstone of many state-of-the-art NLP models due to their ability to handle long-range dependencies in sequential data efficiently.

Transformers architecture consists of various components such as:

1. Transformer block: This consist of Multi Self-attention mechanisms, feed forward network, and layer normalization.
2. Positional encoding: To emphasis the importance of tokens/word in sequence. 
3. Stack of encoder and decoder modules.

<img width="500" alt="image" src="https://github.com/himalayan-avalanche/transformers_nitty_gritty/assets/166877485/89822bbf-5c33-43d7-8245-adaeb3319fd1">


### Attention Mechanism

While RNNs are powerful to learn sequential dependence among data, their use is often limited by the following reasons:
1. Training a RNN network often require large number of hidden states to infer the context in the input sequence. Using large number of hidden states increases the computational complexity, which becomes a bottleneck in training RNNs for applications such as translations, summarization etc.
2. The decoder only uses previous states hidden layer that may not be ideal when it comes to learning from the context of earlier hidden states.

To further illustrate, consider the example of English to French translation:

"The European Economic Zone" -- > "la zone économique européenne". In english there is only one definite article "the". However in French it could corresponds to "le (feminine), la (masculine), les (plural)". If we consider translating English to French word to word, there is no way we could find if we need to translate "the" to "la" as it depends upon other words in the sentence. Therefore using RNNs architecture won't suffice in building a good transaltor. So how can we use the context of other words in the sentence to translate "the" to "la", we need to use some mechanism to do that numerically. This leads to attention mechanism.

To address both of the above limitations in learning long range context, Bahdanau et. al. proposed attention mechanism in their famous paper "Neural Machine Translation by Jointly Learning to Align and Translate (https://arxiv.org/abs/1409.0473)". Using this mechanism, Decoder can use large number of earlier hidden states (context) to generate better prediction.

Lets see how it works. Lets call decoder as the translator. The decoder takes the previous word as input, and the previous hidden states (contexts) to compute which of the hidden states is more relevant to output the next prediction. The decoder assigns a attention score to each of previous hidden states context vectors, and uses a weighted average of the context vectors to use as input context vector to generate next prediction. The more relevant a hidden state the the deocder in predicting next word, higher the attention score of that hidden state. Also worth noting that in translation, when predicting the first word using decoder, the decoder is not using the last input sequence word, instead the special token that indicates the start of a new sequence.

### What are quries, keys and values matrix?

The hidden states of encoder are referred to as Key and Values, whereas the hidden states of decoder is referred to as Values matrix. Of course, the Query, Key and Values matrix go through the affine transformation before computing the attention score, and the context vector. The idea here is something like the hidden states of encoder with affine transformation works as key-value pair, and the decoder's hidden states (queries) query the key value pair to find the values (the context vector is weighted sum of value vector, weights being the alignment score, here is Lilian Wang's great article on this https://lilianweng.github.io/posts/2018-06-24-attention/).

### Encoder, Decoder and Attention


<img width="794" alt="image" src="https://github.com/himalayan-avalanche/transformers_nitty_gritty/assets/166877485/ce3f59e3-c28d-490a-be30-91c5b5ea4aae">



