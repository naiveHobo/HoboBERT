# HoboBERT

Ensemble of 10 modified BERT Small models for Microsoft AI Challenge

### TODO

- Create ensemble model containing 10 copies of BERT Small model
- Each BERT small model will have less encoder layers (probably 6-10)
- Will have to figure out how to identically initialize the trainable 
  layers in all transformer models with the parameters from the corresponding
  layers in the pre-trained BERT Small model.
- Each BERT small model will take as input query-passage pair where the 
  query-passage pair will be preprocessed as defined in originally. Thus,
  each query-passage pair will get its own transformer model.
- The pooled output from each BERT small model will be concatenated into
  one fully-connected layer
- This fully-connected model is then used to decide which query-passage
  pair is the most suitable, using a basic 10 way multi-classification
- The "pooler" converts the encoded sequence tensor of shape
  [batch_size, seq_length, hidden_size] to a tensor of shape
  [batch_size, hidden_size]. The hidden_size used by BERT Small is 768.
  Concatenating 10 x 768 sized layers will create a 7680 sized layer,
  and the fully-connected model would blow up/
- Possible solutions:
  1) Reduce the hidden_size for pooler to some smaller value and
     keep the fully-connected network, thus reducing the number of
     parameters.
  2) Try a transformer model which takes input sequences of size 10,
     where each input sequence is the input which would have otherwise
     gone to the pooler. This may help in taking all sequences into 
     account together. The output of the transformer model, which would
     be reduced in size, can then be fed into a feed-forward network.
     This would still have a lot of parameters, but this may work if the
     number of encoder layers in the transformer model for each individual
     BERT copy is reduced significantly.
  3) Concatenate the pooler output to form a matrix, and use a CNN
     to learn with less parameters. Not exactly sure if this would help
     in the learning in any way, but CNNs have proved to work well
     for NLP tasks in the past.