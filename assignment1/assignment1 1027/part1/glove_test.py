import glove
import numpy as np
import tensorflow as tf
import unittest


class TestGlove(tf.test.TestCase):

    def test_embedding_lookup(self):
        with self.test_session() as sess:
            VOCAB_SIZE = 10
            EMBEDDING_DIM = 11
            wordids = tf.placeholder(tf.int32, shape=[None])
            w, b, embed_matrix = glove.wordids_to_tensors(
                    wordids, EMBEDDING_DIM, VOCAB_SIZE, seed=0)

            sess.run(tf.initialize_all_variables())

            wordids_val = np.array([1, 2, 1])
            w_val, b_val, embed_matrix_val = sess.run(
                    [w, b, embed_matrix], feed_dict={wordids: wordids_val})

            self.assertAllEqual((3, EMBEDDING_DIM), w_val.shape)
            self.assertEquals((3,), b_val.shape)
            self.assertAllEqual(w_val[0], w_val[2])
            self.assertEquals((VOCAB_SIZE, EMBEDDING_DIM), embed_matrix_val.shape)
            # Hope the initializers pick different values for the first element
            # of wid=1 and wid=2.
            self.assertNotEquals(w_val[0][0], w_val[1][0])
            self.assertAllEqual([0, 0, 0], b_val)

    def test_example_weight(self):
        with self.test_session() as sess:
          counts = tf.placeholder(tf.float32, shape=[None])
          weights = glove.example_weight(counts, 100, 0.75)
          weights_val = sess.run(weights, feed_dict={
              counts: np.array([5, 50, 100, 200])})
          self.assertAllClose([0.105737,  0.594604, 1, 1], weights_val)

    def test_loss(self):
        with self.test_session() as sess:
            EMBEDDING_DIM = 3
            w = tf.placeholder(tf.float32, shape=[None, EMBEDDING_DIM])
            b = tf.placeholder(tf.float32, shape=[None])
            w_c = tf.placeholder(tf.float32, shape=[None, EMBEDDING_DIM])
            b_c = tf.placeholder(tf.float32, shape=[None])
            c = tf.placeholder(tf.float32, shape=[None])
            loss = glove.loss(w, b, w_c, b_c, c)

            loss_val = sess.run(loss, feed_dict={
                w: np.array([[1., 2., 3.], [4., 5., 6.]]),
                b: np.array([4., 8.]),
                w_c: np.array([[8., 0, 0], [2., 4., 6.]]),
                b_c: np.array([2., 5.]),
                c: np.array([50, 200])})

            self.assertAllEqual((2,), loss_val.shape)
            self.assertAlmostEquals(60.511177, loss_val[0], places=5)


if __name__ == '__main__':
    unittest.main()
