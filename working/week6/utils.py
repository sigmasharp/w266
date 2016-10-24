import re
import time
import itertools
import numpy as np
import pandas as pd
from IPython.display import display, HTML

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)

##
# Pretty-printing for "search engine" results
def show_search_result(i, doc_id, score, target, link_text):
    link = "[%d] <b><a href=\"%s\" target=\"_blank\">%s</a></b>" % (i, target, link_text)
    info = "document %d (relevance: %.03f)" % (doc_id, score)
    html = link + "\n<br>\n" + info
    display(HTML(html))
    
