# HoboBERT

Ensemble of 10 modified BERT Small models for Microsoft AI Challenge

### Implementation

The first implementation of HoboBERT has the following features:
1. The data is represented as 10 query-passage pairs, where the 
   query in each pair is identical. Each query-pair passage is
   truncated to a sequence length of 64. A special '[CLS]' token
   is inserted before the query and the query and passage sentences
   are separated using a special '[SEP]' token.
2. The HoboBERT model retains the embedding layer fro the original
   BERT model. The embedding layer is universal and is used by each
   individual BERT model in HoboBERT. The embedding layer is initialized
   using the parameters from the pre-trained BERT model.
3. 10 individual BERT models are created, one for each query-passage
   pair. Each BERT model uses the universal embedding layer for sentence
   representation.
4. The encoder in each BERT model is reduced to 3 transformer blocks
   unlike the 12 transformer blocks used in the original BERT model.
   The transformer blocks in each BERT model are initialized identically
   using the parameters from the pre-trained BERT model.
5. The "pooler" layer in each BERT model pools the model by simply taking
   the hidden state corresponding to the first token. This representation
   is followed by a fully-connected layer containing the same the number
   of neurons as the hidden layer size in the transformer blocks.
6. Another dense layer is added after the "pooler" layer in each BERT 
   model to reduce the representation size and thus, the number of 
   parameters further down the model. The size of this dense layer is 128.
7. The output of this layer is taken from each individual BERT model
   and concatenated into a single layer of size 128 * 10 = 1280.
8. This merged layer is followed by another dense layer of size 10
   which is finally used to predict the best passage for the query.

The second implementation of HoboBERT has the following features:
1. The architecture is similar to the first implementation till the pooler
   layers in the individual BERT models.
2. The output of the pooler layers from the BERT models is stacked to 
   create a tensor of size [batch_size, 10, hidden_size].
3. The tensor is used as a sentence sequence and can be fed into another
   encoder containing 3 transformer blocks. This encoder uses attention
   and can thus learn to attend to the encoded sentence that encodes the 
   best query-passage pair.
4. The "merged_pooler" layer takes the hidden state of the first sentence
   in the sequence to pool the merged encoder model.
5. The output of the "merged_pooler" layer is then connected to a dense
   layer which produces the classification scores.

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