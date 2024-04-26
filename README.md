# transformers_nitty_gritty

### Attention Mechanism

While RNNs are powerful to learn sequential dependence among data, their use is often limited by the following reasons:
1. Training a RNN network often require large number of hidden states to infer the context in the input sequence. Using large number of hidden states increases the computational complexity, which becomes a bottleneck in training RNNs for applications such as translations, summarization etc.
2. The decoder only uses previous states hidden layer that may not be ideal when it comes to learning from the context of earlier hidden states.

To further illustrate, consider the example of English to French translation:

"The European Economic Zone" -- > "la zone économique européenne". In english there is only one definite article "the". However in French it could corresponds to "le (feminine), la (masculine), les (plural)". If we consider translating English to French word to word, there is no way we could find if we need to translate "the" to "la" as it depends upon other words in the sentence. Therefore using RNNs architecture won't suffice in building a good transaltor. So how can we use the context of other words in the sentence to translate "the" to "la", we need to use some mechanism to do that numerically. This leads to attention mechanism.

To address both of the above limitations in learning long range context, Bahdanau et. al. proposed attention mechanism in their famous paper "Neural Machine Translation by Jointly Learning to Align and Translate (https://arxiv.org/abs/1409.0473)". Using this mechanism, Decoder can use large number of earlier hidden states (context) to generate better prediction.



