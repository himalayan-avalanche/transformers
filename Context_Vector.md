# Whats context vector?

If you look at RNN or transformer based architecture, its often based on Encoder-Decoder archiecture (and lot in between). The Encoder and Decoder is often stack of RNN cells, 
or multi head self attention blocks used to generate the context vector. The context vector along with hidden state output concatenated and then passed through a linear layer to generate the vector same size as vocabulary.
This vocab size vector is then passed to a softmax function to generate the probabilities vector, which in turn is then used to find most likely word/token prediction.
