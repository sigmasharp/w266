import tensorflow as tf

def wordids_to_tensors(wordids, embedding_dim, vocab_size, seed=0):
    '''Convert a list of wordids into embeddings and biases.

    This function creates a variable for the embedding matrix, dimension |E x V|
    and a variable to hold the biases, dimension |V|.

    It returns an op that will accept the output of "wordids" op and lookup
    the corresponding embedding vector and bias in the table.

    Args:
      - wordids |IDs|: a tensor of wordids
      - embedding_dim, |E|: a scalar value of the # of dimensions in which to embed words
      - vocab_size |V|: # of terms in the vocabulary

    Returns:
      A tuple (ws, bs, m), all with tf.float32s:
        - ws |IDs x E| is a tensor of word embeddings
        - bs |IDs| is a vector of biases.
        - m |V x E| is the full embedding matrix (from which you looked up ws)
          (we don't strictly need this, but it comes in handy in the "Play!" section.)

    HINT: To get the tests to pass, initialize w with a random_uniform [-1, 1] using
          the provided seed.  As usual the "b" vector should be initialized to 0.

    HINT: Look at tf.embedding_lookup(.).
    '''
    # START YOUR CODE
    # tf.reset_default_graph
    # IDs = wordids.get_shape()[0]
    # print IDs
    E = embedding_dim
    V = vocab_size
    sh = [wordids.get_shape()[0], E]
    #ws = tf.random_uniform([wordids.get_shape()[0], [E]] , minval=-1.0, maxval=1.0, dtype=tf.float32, seed=seed)
    ws = tf.Variable(tf.random_uniform(sh, [E]), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=seed), validate_shape=False, name="ws")
    bs = tf.Variable(tf.zeros(wordids.get_shape()), dtype=tf.float32, validate_shape=False, name="b")
    m  = tf.nn.embedding_lookup(params=(ws), ids=wordids, name="m")
    return (ws, bs, m)
    # END YOUR CODE


def example_weight(Xij, x_max, alpha):
    '''Scale the count according to Equation (9) in the Glove paper.

    This runs as part of the TensorFlow graph.  You must do this with
    TensorFlow ops.

    Args:
      - Xij: a |batch| tensor of counts.
      - x_max: a scalar, see paper.
      - alpha: a scalar, see paper.

    Returns:
      - A vector of corresponding weights.
    '''
    # START YOUR CODE
    pass
    # END YOUR CODE


def loss(w, b, w_c, b_c, c):
    '''Compute the loss for each of training examples.

    Args:
      - w |batch_size x embedding_dim|: word vectors for the batch
      - b |batch_size|: biases for these words
      - w_c |batch_size x embedding_dim|: context word vectors for the batch
      - b_c |batch_size|: biases for context words
      - c |batch_size|: # of times context word appeared in context of word

    Returns:
      - loss |batch_size|: the loss of each example in the batch
    '''
    # START YOUR CODE
    pass
    # END YOUR CODE
