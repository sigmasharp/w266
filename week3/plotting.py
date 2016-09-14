import os
from IPython.display import display, HTML
import bokeh.plotting as bp


def save_bokeh_plot(plot, title="", filename=None, basedir="plots"):
  from bokeh.embed import file_html
  from bokeh.resources import CDN
  import uuid
  
  if filename is None:
    # Generate random hex string
    name = uuid.uuid4().hex[:6] + ".html"
    filename = os.path.join(basedir, name)

  savedir = os.path.dirname(filename)
  if not os.path.isdir(savedir):
    os.makedirs(savedir)
  
  html = file_html(plot, CDN, title)
  with open(filename, 'w') as fd:
    fd.write(html)
  return filename

def plot_wv(Wv, vocab, num_words=1000, title="Word Vectors", 
	    word_colors=None, word_sizes=None,
	    inline=True, filename=None, **text_kw):
  """Make an interactive plot of the (first two components) 
  of the word vectors."""
  top_counts = vocab.unigram_counts.most_common(num_words)
  selected_ids = vocab.words_to_ids(w for w,c in top_counts)
  selected_words = vocab.ids_to_words(selected_ids)

  px = Wv[selected_ids, 0]
  py = Wv[selected_ids, 1]
  
  p = bp.figure(title="Word Vectors", 
		tools="reset,pan,wheel_zoom,box_zoom", 
		active_scroll="wheel_zoom")
  text_kwargs = dict(text_baseline='middle',
		     text_align='center')
  if word_colors is not None:
    text_kwargs['text_color'] = [word_colors[w] for w in selected_words]
  if word_sizes is not None:
    text_kwargs['text_font_size'] = [word_sizes[w] for w in selected_words]
  text_kwargs.update(text_kw)  # override anything
  p.text(px, py, text=selected_words, **text_kwargs)
  
  if inline:
    bp.show(p)
  else:
    p.plot_width = 1200
    p.plot_height = 900
    filename = save_bokeh_plot(p, title=title, filename=filename)
    link_text = "View plot \"%s\" in a new tab" % filename
    display(HTML("<a href=\"{filename}\" target='_blank'>{link_text}</a>".format(**locals())))
