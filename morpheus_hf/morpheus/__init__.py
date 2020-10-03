from .morpheus_qa import MorpheusHuggingfaceQA
from .morpheus_nli import MorpheusHuggingfaceNLI
from .morpheus_seq2seq import MorpheusHuggingfaceSummarization

from nltk import download
download('averaged_perceptron_tagger')
download('universal_tagset')
