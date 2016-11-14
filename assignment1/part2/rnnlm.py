import time

import tensorflow as tf
import numpy as np

def matmul3d(X, W):
  """Wrapper for tf.matmul to handle a 3D input tensor X.
  Will perform multiplication along the last dimension.

  Args:
    X: [m,n,k]
    W: [k,l]

  Returns:
    XW: [m,n,l]
  """
  Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
  XWr = tf.matmul(Xr, W)
  newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
  return tf.reshape(XWr, newshape)

def MakeFancyRNNCell(H, keep_prob, num_layers=1):
  """Make a fancy RNN cell.

  Use tf.nn.rnn_cell functions to construct an LSTM cell.
  Initialize forget_bias=0.0 for better training.

  Args:
    H: hidden state size
    keep_prob: dropout keep prob (same for input and output)
    num_layers: number of cell layers

  Returns:
    (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
  """
  #### YOUR CODE HERE ####
  cell = None  # replace with something better

  # Solution
  cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
  cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                       output_keep_prob=keep_prob)
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

  #### END(YOUR CODE) ####
  return cell

class RNNLM(object):

  def __init__(self, V, H, num_layers=1):
    """Init function.

    This function just stores hyperparameters. You'll do all the real graph
    construction in the Build*Graph() functions below.

    Args:
      V: vocabulary size
      H: hidden state dimension
      num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
    """
    # Model structure; these need to be fixed for a given model.
    self.V = V
    self.H = H
    self.num_layers = num_layers #*** took out the 1 and replaced it with num_layers

    # Training hyperparameters; these can be changed with feed_dict,
    # and you may want to do so during training.
    with tf.name_scope("Training_Parameters"):
      self.learning_rate_ = tf.constant(0.1, name="learning_rate")
      #self.dropout_keep_prob_ = tf.constant(0.5, name="dropout_keep_prob")
      self.dropout_keep_prob_ = tf.placeholder(tf.float32, name="dropout_keep_prob")
      
      # For gradient clipping, if you use it.
      # Due to a bug in TensorFlow, this needs to be an ordinary python
      # constant instead of a tf.constant.
      self.max_grad_norm_ = 5.0
    # ==>
    # data members: all by 'self.' prefix and all are hyper parameters for the learning for the model
    # V, H, num_layers, learning_rate_, dropout_keep_prob_, max_grad_norm_

  def BuildCoreGraph(self):
    """Construct the core RNNLM graph, needed for any use of the model.

    This should include:
    - Placeholders for input tensors (input_w_, initial_h_, target_y_)
    - Variables for model parameters
    - Tensors representing various intermediate states
    - A Tensor for the final state (final_h_)
    - A Tensor for the output logits (logits_), i.e. the un-normalized argument
      of the softmax(...) function in the output layer.
    - A scalar loss function (loss_)

    Your loss function should return a *scalar* value that represents the
    _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
    tf.reduce_mean).

    You shouldn't include training or sampling functions here; you'll do this
    in BuildTrainGraph and BuildSampleGraph below.

    We give you some starter definitions for input_w_ and target_y_, as well
    as a few other tensors that might help. We've also added dummy values for
    initial_h_, logits_, and loss_ - you should re-define these in your code as
    the appropriate tensors. See the in-line comments for more detail.
    """
    # Input ids, with dynamic shape depending on input.
    # Should be shape [batch_size, max_time] and contain integer word indices.
    self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

    # Initial hidden state. You'll need to overwrite this with cell.zero_state
    # once you construct your RNN cell.
    
    self.initial_h_ = None

    # Final hidden state. You'll need to overwrite this with the output from
    # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
    # applicable).
    
    # ==> the variable
    self.final_h_ = None

    # Output logits, which can be used by loss functions or for prediction.
    # Overwrite this with an actual Tensor of shape [batch_size, max_time]
    
    # =>=>=>
    # logits = o (or h) x Wout + b, where o is of size 1xH, Wout is HxV or Hxk, and b is V
    self.logits_ = None

    # Should be the same shape as inputs_w_
    self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

    # Replace this with an actual loss function
    #self.loss_ = tf.reduced_sum(tf.nn.softmax(self.logits_, dim=-1, name=None))
    self.loss_ = None

    # Get dynamic shape info from inputs
    with tf.name_scope("batch_size"):
      self.batch_size_ = tf.shape(self.input_w_)[0]
    with tf.name_scope("max_time"):
      self.max_time_ = tf.shape(self.input_w_)[1]

    # Get sequence length from input_w_.
    # This will be a vector with elements ns[i] = len(input_w_[i])
    # You can override this in feed_dict if you want to have different-length
    # sequences in the same batch, although you shouldn't need to for this
    # assignment.
    self.ns_ = tf.tile([self.max_time_], [self.batch_size_,], name="ns")

    #### YOUR CODE HERE ####
    
    # seed
    self.seed = 0

    # Construct embedding layer
    # from V x H to H
    
    # notation: i am going to use b for batch_size, and t for max_time in the comments
    
    # Wem of shape(V, H), the embedding layer weight matrix, or the lookup table
    self.Wem_ = tf.Variable(tf.random_uniform([self.V, self.H], minval=-1.0, maxval=1.0, seed=0), name="Wem")
    
    # no bem, not like part 1, which was there because it participates on the loss_ for the embedding learning, now only em_W
    # is needed for the overall RNNML learningN
    # self.bem_ = tf.Variable(tf.zeros(self.V), dtype=tf.float32, name="bem_")
    
    # the input_x_ of shape(b, t, H), the input vector, from the lookup of intput_w_ of the shape [b, t], 
    # an index int between 0 and V - 1 against the lookup table Wem,
    # i don't think the inner reshape to the input_w_ is necessary, but i did it anyway
    self.input_x_ = tf.reshape(tf.nn.embedding_lookup(params=(self.Wem_), 
        ids=self.input_w_), [self.batch_size_, self.max_time_, self.H], name="x")
    
    # Construct RNN/LSTM cell and recurrent layer (hint: use tf.nn.dynamic_rnn)
    
    # create the fancy cell, a LSTM cell (with 4 affine layers)
    self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
    
    # initial_h_ of shape(b, t, H), supposedly, the initial hidden state for the rnn layer
    #self.initial_h_ = tf.placeholder(tf.float32, [None, None, self.H], name="inital_h")
    # initialize it to all zeros at the beginning
    #print self.initial_h_.get_shape() 
    self.initial_h_ = self.cell_.zero_state(self.batch_size_, dtype=tf.float32)

    # ouput of shape(b, t, H), the output state or vector from the rnn layer, or the input to the output layer
    # final_h of shape(b, t, H), the final state from the rnn layer
    self.output_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, inputs=self.input_x_, 
        sequence_length=self.ns_, initial_state=self.initial_h_, dtype=tf.float32)
   

    # Softmax output layer, over vocabulary
    # Hint: use the matmul3d() helper here.
    
    # Wout of shape(H, V), the output layer weight matrix, and 
    # bout of shape(V), the output layer bias vector
    self.Wout_ = tf.Variable(tf.random_uniform([self.H, self.V], minval=-1.0, maxval=1.0, seed=self.seed), name="Wout")
    self.bout_ = tf.Variable(tf.zeros([self.V]), dtype=tf.float32, name="b")
    
    # logits of shape(b, t, H), the logits from the output layer, for the whole RNNLM
    self.logits_ = tf.reshape(matmul3d(self.output_, self.Wout_) + self.bout_, [self.batch_size_, self.max_time_, self.V])
    #print self.logits_.get_shape()
    
    # y^hat of shape(b, t), the softmax from the logits
    self.y_hat_ = tf.reshape(tf.argmax(tf.reshape(self.logits_, [-1, self.V]), 1, name="y_hat"), [self.batch_size_, self.max_time_])
    #print self.y_hat_.get_shape()
    
    # Loss computation (true loss, for prediction)
    # loss of shape (), a scalar of the true loss, or the sum of the cross entrypy loss over the logits
    self.loss_ = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_, self.target_y_, name = "loss"))    

    #### END(YOUR CODE) ####


  def BuildTrainGraph(self):
    """Construct the training ops.

    You should define:
    - train_loss_ (optional): an approximate loss function for training
    - train_step_ : a training op that can be called once per batch

    Your loss function should return a *scalar* value that represents the
    _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
    tf.reduce_mean).
    """
    # Replace this with an actual training op
    self.train_step_ = tf.no_op(name="dummy")

    # Replace this with an actual loss function
    self.train_loss_ = None

    #### YOUR CODE HERE ####

    # Define loss function(s)
    with tf.name_scope("Train_Loss"):
      # Placeholder: replace with a sampled loss
    
      #  train_loss of shape(), the sum of a sampled softmax loss over the sampled 
      self.num_sampled = 100
      self.train_loss_ = tf.reduce_sum(tf.nn.sampled_softmax_loss(tf.transpose(self.Wout_), self.bout_, 
          tf.reshape(self.output_, [-1, self.H]), labels=tf.reshape(self.target_y_, [-1,1]), num_sampled=self.num_sampled, 
          num_classes=self.V, name='train_loss'))
      #self.train_loss_ = self.loss_
      

    # Define optimizer and training op
    with tf.name_scope("Training"):
        self.optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
        self.train_step_ = self.optimizer_.minimize(self.train_loss_)
    
    # Initializer step: done explicitly in c. run training
    # init_ = tf.initialize_all_variables()

    #### END(YOUR CODE) ####


  def BuildSamplerGraph(self):
    """Construct the sampling ops.

    You should define pred_samples_ to be a Tensor of integer indices for
    sampled predictions for each batch element, at each timestep.

    Hint: use tf.multinomial, along with a couple of calls to tf.reshape
    """
    # Replace with a Tensor of shape [batch_size, max_time, 1]
    self.pred_samples_ = None

    #### YOUR CODE HERE ####
    #self.pred_proba_ = tf.nn.softmax(self.logits_, name="pred_proba")
    #self.pred_max_ = tf.argmax( self. logits_, 1, name="pred_max")

    self.pred_samples_ = tf.reshape(tf.multinomial(tf.reshape(self.logits_, [-1, self.V]), 1), [self.batch_size_, self.max_time_, 1])
    

    #### END(YOUR CODE) ####

