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

def BuildHistogram(filename='english_uni_simplified_sorted_top'):
  histo = {}
  for line in open(filename):
    result = re.match('"(.*)" (.*)', line)
    histo[result.group(1).lower()] = (histo.get(result.group(1).lower(), 0) +
                                      int(result.group(2)))
  return histo 
