import numpy as np
import tensorflow as tf

class AddTwo(object):
    def __init__(self):
        # If you are constructing more than one graph within a Python kernel
        # you can either tf.reset_default_graph() each time, or you can
        # instantiate a tf.Graph() object and construct the graph within it.

        # Hint: Recall from live sessions that TensorFlow
        # splits its models into two chunks of code:
        # - construct and keep around a graph of ops
        # - execute ops in the graph
        #
        # Construct your graph in __init__ and run the ops in Add.
        #
        # We make the separation explicit in this first subpart to
        # drive the point home.  Usually you will just do them all
        # in one place, including throughout the rest of this assignment.
        #
        # Hint:  You'll want to look at tf.placeholder and sess.run.

        # START YOUR CODE
        self.graph = tf.Graph()
        self.sess = tf.Session()
        self.x0 = tf.placeholder(tf.float32, None)
        self.y0 = tf.placeholder(tf.float32, None)

        
        # END YOUR CODE

    def Add(self, x, y):
        # START YOUR CODE
        r = tf.add(self.x0, self.y0)
        return(self.sess.run(r, feed_dict={self.x0: x, self.y0: y})) 
        # END YOUR CODE

def affine_layer(hidden_dim, x, seed=0):
    # x: a [batch_size x # features] shaped tensor.
    # hidden_dim: a scalar representing the # of nodes.
    # seed: use this seed for xavier initialization.

    # START YOUR CODE
    #np.random.seed([seed])
    
    b_size = x.get_shape()[0]
    f_size = x.get_shape()[1]
    
    #print x.get_shape(), b_size, f_size, hidden_dim
    #W = tf.Variable(tf.zeros([f_size, hidden_dim]), dtype=tf.float32, name="W")

    W = tf.get_variable(name = "W", shape=[f_size, hidden_dim], initializer=tf.contrib.layers.xavier_initializer(seed=seed))

    b = tf.Variable(tf.zeros([hidden_dim]), dtype=tf.float32, name="b")

    return(tf.matmul(x, W) + b)
    # END YOUR CODE

def fully_connected_layers(hidden_dims, x):
    # hidden_dims: A list of the width of the hidden layer.
    # x: the initial input with arbitrary dimension.
    # To get the tests to pass, you must use relu(.) as your element-wise nonlinearity.
    #
    # Hint: see tf.variable_scope - you'll want to use this to make each layer 
    # unique.
    # Hint: a fully connected layer is a nonlinearity of an affine of its input.
    #       your answer here only be a couple of lines long (mine is 4).

    # START YOUR CODE
    if len(hidden_dims) == 0:
        return(affine_layer(1, x))
    z = tf.nn.relu(affine_layer(hidden_dims[0], x))
    for i in hidden_dims[1:]:
        with tf.variable_scope("scope"+str(i)):
            z = tf.nn.relu(affine_layer(i, z))
    return(z)
        
    # END YOUR CODE

def train_nn(X, y, X_test, hidden_dims, batch_size, num_epochs, learning_rate,
             verbose=False):
    # Train a neural network consisting of fully_connected_layers
    # to predict y.  Use sigmoid_cross_entropy_with_logits loss between the
    # prediction and the label.
    # Return the predictions for X_test.
    # X: train features
    # y: train labels
    # X_test: test features
    # hidden_dims: same as in fully_connected_layers
    # learning_rate: the learning rate for your GradientDescentOptimizer.

    # Construct the placeholders.
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=[None, X.shape[-1]])
    y_ph = tf.placeholder(tf.float32, shape=[None])
    global_step = tf.Variable(0, trainable=False)

    # Construct the neural network, store the batch loss in a variable called `loss`.
    # At the end of this block, you'll want to have these ops:
    # - y_hat: probability of the positive class
    # - loss: the average cross entropy loss across the batch
    #   (hint: see tf.sigmoid_cross_entropy_with_logits)
    #   (hint 2: see tf.reduce_mean)
    # - train_op: the training operation resulting from minimizing the loss
    #             with a GradientDescentOptimizer
    #
    # Hint:  Remember that a neural network has the form
    #        <affine-nonlinear>* -> affine -> sigmoid -> y_hat
    #        Double check your code works for 0..n affine-nonlinears.
    #
    # START YOUR CODE
    logits = fully_connected_layers(hidden_dims, x_ph)
    y_hat = tf.sigmoid(logits)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_hat, name="loss"))
    

    alpha = learning_rate
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_step = optimizer.minimize(loss)
        
    init = tf.initialize_all_variables()
    #with tf.variable_scope("here"):
    #    y_pred = tf.sigmoid(fully_connected_layers(hidden_dims, x_ph)) >= 0.5
    # END YOUR CODE


    # Output some initial statistics.
    # You should see about a 0.6 initial loss (-ln 2).
    sess = tf.Session(config=tf.ConfigProto(device_filters="/cpu:0"))
    sess.run(init)
    print 'Initial loss:', sess.run(loss, feed_dict={x_ph: X, y_ph: y})

    if verbose:
      for var in tf.trainable_variables():
          print 'Variable: ', var.name, var.get_shape()
          print 'dJ/dVar: ', sess.run(
                  tf.gradients(loss, var), feed_dict={x_ph: X, y_ph: y})

    for epoch_num in xrange(num_epochs):
        for batch in xrange(0, X.shape[0], batch_size):
            X_batch = X[batch : batch + batch_size]
            y_batch = y[batch : batch + batch_size]
            # Feed a batch to your network using sess.run.
            # Populate loss_value with the current value of loss.
            # Populate global_value with the current value of global_step.
            # You'll also want to run your training op.
            # START YOUR CODE
            
            #y_pred = sess.run(y_hat, feed_dict={x_ph: X_batch, y_ph: y_batch})

            # Run a single gradient descent step
            loss_value, y_prob, _ = sess.run([loss, y_hat, train_step], feed_dict={x_ph: X, y_ph: y})
            global_step += 1
           

            # END YOUR CODE
        if epoch_num % 300 == 0:
            print 'Step: ', global_step, 'Loss:', loss_value
            if verbose:
              for var in tf.trainable_variables():
                  print var.name, sess.run(var)
              print ''

    # Return your predictions.
    # START YOUR CODE
    # plug in X_test
    #y_pred = session.run(y_hat, 
    return sess.run(y_pred)
    # END YOUR CODE
