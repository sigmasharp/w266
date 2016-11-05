import word_utils

import unittest
class TestWordStreams(unittest.TestCase):

    def test_vocab(self):
        vocab = word_utils.Vocabulary('hello hello world world lonely'.split(), 2)
        self.assertEquals(1, vocab.to_id('hello'))
        self.assertEquals(2, vocab.to_id('world'))
        self.assertEquals(0, vocab.to_id('lonely'))
        self.assertEquals(0, vocab.to_id('something'))

        self.assertEquals('hello', vocab.to_word(1))
        self.assertEquals('world', vocab.to_word(2))
        self.assertEquals('<UNK>', vocab.to_word(0))
        self.assertEquals('<UNK>', vocab.to_word(3))

    def test_group_prefix(self):
        self.assertEquals(
                set([('a',),
                     ('a', 'b')]),
                word_utils.group_prefix([('a', 'b', 'c')]))

    def test_grouped_stream(self):
        words = 'hello world the new york times'.split()
        grouped = 'hello_world the_new_york_times'.split()
        self.assertEquals(grouped,
                word_utils.grouped_stream(
                    words, [
                        tuple('the new york times'.split()),
                        tuple('hello world'.split())
                    ]))

    def test_unigram_and_bigram_counts(self):
        uni, bi = word_utils.unigram_and_bigram_counts('hello hello world'.split())

        self.assertEquals({'hello': 2, 'world': 1}, uni)
        self.assertEquals({('hello', 'hello'): 1, ('hello', 'world'): 1}, bi)


if __name__ == '__main__':
    unittest.main()
