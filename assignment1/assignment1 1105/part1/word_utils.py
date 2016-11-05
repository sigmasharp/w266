import collections

class Vocabulary(object):

    def __init__(self, words, size):
        word_counts = [(count, word) for word, count in collections.Counter(words).iteritems()]
        top_words = zip(*(sorted(word_counts, reverse=True)[:size]))[1]
        all_words = ['<UNK>'] + sorted(top_words)
        self.word_to_id = {word: wid for wid, word in enumerate(all_words)}
        self.id_to_word = {wid: word for word, wid in self.word_to_id.iteritems()}

    def to_id(self, word):
        return self.word_to_id.get(word, 0)

    def to_word(self, wid):
        return self.id_to_word.get(wid, '<UNK>')

    def size(self):
        return len(self.word_to_id)


def group_prefix(groups):
    group_prefix = set()
    for group in groups:
        for i in xrange(len(group) - 1):
            group_prefix.add(group[0 : i + 1])

    return group_prefix


def grouped_stream(words, groups):
    groups = set(groups)
    prefixes = group_prefix(groups)

    output = []
    state = []
    for i, word in enumerate(words):
        state.append(word)
        while state:
          state_tuple = tuple(state)
          if i + 1 < len(words) and state_tuple in prefixes:
              break 
          if state_tuple in groups:
              output.append('_'.join(state_tuple))
              state = []
              break 
          output.append(state[0])
          state = state[1:]
    return output


def unigram_and_bigram_counts(words):
    unigram_counts = collections.Counter(words)
    bigram_counts = collections.Counter(zip(words[0:-1], words[1:]))
    return unigram_counts, bigram_counts


