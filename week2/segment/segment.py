import re
import math

def memo(fn):
  cache = {}
  def docache(arg):
    if arg in cache:
      return cache[arg]
    val = fn(arg)
    cache[arg] = val
    return val
  return docache

@memo
def segment(text):
  if not text: return []
  candidates = ([first]+segment(rem) for first, rem in splits(text))
  return max(candidates, key=Pwords)

def splits(text, L=20):
  return [(text[:i+1], text[i+1:])
          for i in range(min(len(text), L))]

histo = {}
def Pw(w):
  totals = histo['']
  if w in histo:  return math.log(histo[w]) - math.log(totals)
  else: return -math.log(totals) - 3*len(w)

def Pwords(words):
  return sum(Pw(w) for w in words)

for line in open('english_uni_simplified_sorted_top'):
  result = re.match('"(.*)" (.*)', line)
  histo[result.group(1).lower()] = (histo.get(result.group(1).lower(), 0) +
                                    int(result.group(2)))

while True:
  text = raw_input('>')
  print segment(text.lower())
