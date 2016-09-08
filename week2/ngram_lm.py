from collections import defaultdict
import numpy as np

class SmoothedTrigramLM(object):
  """Smoothed trigram language model.
 
  Implements smoothing lazily at predict time, which isn't particularly 
  efficient, but does allow testing of different hyperparameters without 
  recomputing counts of the corpus.
  """ 
 
  def __init__(self, words):
    # Raw counts over the corpus. 
    # Keys are context (N-1)-grams, values are dicts of word -> count.
    # You can access C(w | w_{i-1}, ...) as:
    # unigram: self.counts[()][w]
    # bigram:  self.counts[(w_1,)][w]
    # trigram: self.counts[(w_2,w_1)][w]
    self.counts = defaultdict(lambda: defaultdict(lambda: 0))
    # Context types: store the set of preceding words for each word
    self.type_contexts = defaultdict(lambda: set())
    
    # Iterate through the word stream once
    # Compute unigram, bigram, trigram counts and type fertilities
    # Unigram: C(word)
    # Bigram: C(word | b)
    # Trigram: C(word | b a)
    w_1, w_2 = None, None
    for word in words:
      self.counts[()][word] += 1
      if w_1 is not None:
        self.counts[(w_1,)][word] += 1
        self.type_contexts[word].add(w_1)
        if w_2 is not None:
          self.counts[(w_2,w_1)][word] += 1
      # Update context
      w_2 = w_1
      w_1 = word
    
    ##
    # We'll compute type fertilities and normalization constants now,
    # but not actually store the normalized probabilities. That way, we can compute
    # them (efficiently) on the fly.

    # Compute type fertilities, and sum of them (call this z_kn).
    self.type_fertility = {w:len(s) for w,s in self.type_contexts.iteritems()}
    self.z_kn = float(sum(self.type_fertility.values()))
    # Count the total for each context.
    self.context_total = {k:float(sum(c.values())) for k,c in self.counts.iteritems()}
    # Count the number of nonzero entries for each context.
    self.context_nnz = {k:len(c) for k,c in self.counts.iteritems()}
    # Count total vocab size, from unigram counts.
    self.vocab_size = len(self.counts[()])
    
    # Make a static copy of the word list, for prediction.
    self.words = self.counts[()].keys()
  
  ##
  # Implement add-k smoothing, discounting, interpolation, and backoff
  def get_proba(self, word, context, add_k=0.0):
    c0 = self.counts[context][word]
    z0 = self.context_total.get(context, 0)
    denom = (z0 + add_k * self.vocab_size)
    if denom == 0: return 0
    return (c0 + add_k) / denom
    
  def next_word_proba_basic(self, word, seq, interpolation=None, 
            backoff=False, add_k=0.0, **kw):
    if interpolation is not None:
      l1,l2,l3 = interpolation
      pw = (l1*self.get_proba(word, (), add_k=add_k) + 
            l2*self.get_proba(word, tuple(seq[-1:]), add_k=add_k) + 
            l3*self.get_proba(word, tuple(seq[-2:]), add_k=add_k))
      return pw
      
    if backoff > 0:
      # Try trigram, then bigram, then unigram
      for context in (tuple(seq[-2:]), tuple(seq[-1:]), ()):
        if self.context_total.get(context, 0) >= backoff:
          return self.get_proba(word, context, add_k=add_k)
      # Return smoothed unigram probability, if all else fails
      return self.get_proba(word, (), add_k=add_k)
    else:
      # Smoothed trigram probability
      return self.get_proba(word, tuple(seq[-2:]), add_k=add_k)
    
  ##
  # Implement KN smoothing  
  def kn_interp(self, word, context, delta, pw):
    """Compute KN estimate by recursive backoff / interpolation."""
    c0 = self.counts[context][word]
    z0 = self.context_total.get(context, 0)
    # If context is never seen, pass through backoff unchanged
    if z0 == 0: return pw
    # lc is lambda_context: normalize interpolation based on
    # total mass removed by discounting
    lc = delta * (self.context_nnz.get(context, 0) / z0)
    pwc = max(0, c0 - delta)/z0
    return pwc + lc * pw
    
  def next_word_proba_kn(self, word, seq, delta=0.75, **kw):
    """Compute next word probability with KN backoff smoothing."""
    # KN unigram, then recursively compute bigram, trigram
    pw1 = self.type_fertility[word] / self.z_kn
    pw2 = self.kn_interp(word, tuple(seq[-1:]), delta, pw1)
    pw3 = self.kn_interp(word, tuple(seq[-2:]), delta, pw2)
    return pw3
  
  ##
  # Standard LM functions
  def next_word_proba(self, word, seq, kn=True, **kw):
    if kn: return self.next_word_proba_kn(word, seq, **kw)
    else: return self.next_word_proba_basic(word, seq, **kw)
  
  def predict_next(self, seq, **kw):
    # Sample a word from the conditional distribution
    probs = [self.next_word_proba(word, seq, **kw) for word in self.words]
    return np.random.choice(self.words, p=probs)
  
  def score_seq(self, seq, verbose=False, **kw):
    # Score all trigrams, so we start at position 2
    score = 0.0
    for i in range(2, len(seq)):
      s = np.log2(self.next_word_proba(seq[i], seq[:i], **kw))
      score += s
      # DEBUG
      if verbose:
        print "log P(%s | %s) = %.03f" % (seq[i], " ".join(seq[i-2:i]), s)
    return score
