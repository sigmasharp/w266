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
    self.num_layers = num_layers #took out the 1 and replaced it with num_layers

    # Training hyperparameters; these can be changed with feed_dict,
    # and you may want to do so during training.
    with tf.name_scope("Training_Parameters"):
      self.learning_rate_ = tf.constant(0.1, name="learning_rate")
      self.dropout_keep_prob_ = tf.constant(0.5, name="dropout_keep_prob")
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
    
    # ==> "ONCE?"=> when
    self.initial_h_ = None

    # Final hidden state. You'll need to overwrite this with the output from
    # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
    # applicable).
    
    # ==> the variable
    self.final_h_ = tf.placeholder(tf.float32, [None, None, self.H], name = "final_h")

    # Output logits, which can be used by loss functions or for prediction.
    # Overwrite this with an actual Tensor of shape [batch_size, max_time]
    
    # =>=>=>
    # logits = o (or h) x Wout + b, where o is of size 1xH, Wout is HxV or Hxk, and b is V
    self.logits_ = tf.placeholder(tf.float32, [None, None, self.H], name = "logits")

    # Should be the same shape as inputs_w_
    self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

    # Replace this with an actual loss function
    #self.loss_ = tf.reduced_sum(tf.nn.softmax(self.logits_, dim=-1, name=None))
    self.loss_ = tf.placeholder(tf.int32, [None, None, self.H], name="loss")

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

    # Construct embedding layer
    # from V x H to H
    self.em_mat_ = tf.Variable(tf.random_uniform([self.V, self.H], minval=-1.0, maxval=1.0, seed=0), name="em_mat")
    self.em_b_ = tf.Variable(tf.zeros(self.V), dtype=tf.float32, name="em_b")
    self.em_lu_ = tf.reshape(tf.nn.embedding_lookup(params=(self.em_mat_), ids=tf.reshape(self.input_w_, [-1])), [self.batch_size_, self.max_time_, self.H], name="em_lu")
    
    self.em_lu_b = tf.reshape(tf.nn.embedding_lookup(params=(self.em_b_), ids=tf.reshape(self.input_w_, [-1])), [self.batch_size_, self.max_time_, self.H], name="em_lu_b")

    # Construct RNN/LSTM cell and recurrent layer (hint: use tf.nn.dynamic_rnn)
    # from 2*H to H
    self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
    
    #self.inital_h_ = tf.reshape(self.cell_.zero_state(self.batch_size_ * self.H, dtype=tf.float32), [self.batch_size_, self.H])
    self.inital_h_ = self.cell_.zero_state(self.batch_size_, dtype=tf.float32)
    #=> tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)
    self.rnn_output_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, inputs=self.em_lu_, sequence_length=self.ns_, initial_state=self.initial_h_, dtype=tf.float32)

    # Softmax output layer, over vocabulary
    # Hint: use the matmul3d() helper here.
    
    # => matmul3d(), [batch, max_time, H] x [H, V]

    # => output = p^hat (w(i+1)) = softmax(o(i)Wout+bout)
    # => tf.nn.softmax(logits, name=None)
    self.Wout_ = tf.Variable(tf.random_uniform([self.H, self.V], minval=-1.0, maxval=1.0, seed=0), name="Wout")
    self.bout_ = tf.Variable(tf.random_uniform([self.V], minval=-1.0, maxval=1.0, seed=0), name="bout")
    self.logits_ = matmul3d(self.rnn_output_, self.Wout_) + self.bout_
    self.output_ = tf.nn.softmax(self.logits_, name="output")
    #print self.logits_.get_shape()
    #print self.output_.get_shape()
    
    # Loss computation (true loss, for prediction)
    # => tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
    self.loss_ = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_, self.target_y_, name = "loss"))
    #print self.loss_.get_shape()
    

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
      self.train_loss_ = tf.reduce_sum(tf.nn.sampled_softmax_loss(tf.transpose(self.Wout_), self.bout_, tf.reshape(self.rnn_output_, [-1, self.H]), labels=self.target_y_, num_sampled=5000, num_classes=self.V, num_true=5000, sampled_values=None, remove_accidental_hits=True, partition_strategy='mod', name='train_loss'))


    # Define optimizer and training op
    with tf.name_scope("Training"):
        self.train_step_ = None  # Placeholder: replace with an actual op
        self.alpha_ = tf.placeholder(tf.float32, name="learning_rate")
        self.optimizer_ = tf.train.AdagradOptimizer(self.alpha_)
        self.train_step_ = self.optimizer_.minimize(self.train_loss_)

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

    self.pred_samples_ = tf.multinomial(tf.reshape(self.logits_, [-1, self.V]), 5000, name="pred_samples")

    #### END(YOUR CODE) ####

