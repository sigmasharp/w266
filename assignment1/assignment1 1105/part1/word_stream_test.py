import word_stream

import unittest
class TestWordStreams(unittest.TestCase):

    def test_context_windows(self):
        words = 'hello how are you ?'.split()
        self.assertEquals([w for w in word_stream.context_windows(words, 5)], [words])

        words = 'hello how are you ? i am in w266 nlp'.split()
        self.assertEquals([w for w in word_stream.context_windows(words, 9)], [words[0:-1], words[1:]])

    def test_cooccurrence_table(self):
        table = sorted(word_stream.cooccurrence_table('hello world how are you today'.split(), C=2))
        expected = sorted([
            ('how', 'are', 1.0),
            ('how', 'hello', 0.5),
            ('how', 'world', 1.0),
            ('how', 'you', 0.5),
            ('are', 'world', 0.5),
            ('are', 'how', 1.0),
            ('are', 'you', 1.0),
            ('are', 'today', 0.5),
        ])
        self.assertEquals(table, expected)

    def test_score_bigram(self):
        unigrams = {'hello': 5, 'world': 3}
        bigrams = {('hello', 'world'): 8}

        score = word_stream.score_bigram(('hello', 'world'), unigrams, bigrams, 1)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.466666667)

        score = word_stream.score_bigram(('OOV', 'world'), unigrams, bigrams, 1)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()
