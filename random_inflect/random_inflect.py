from mosestokenizer import MosesTokenizer, MosesDetokenizer
import random, nltk, lemminflect

'''
inflection_counts should have the following structure:
{ PTB tag: int, ... , PTB tag: int }
'''
def random_inflect(source: str, inflection_counts: Dict[str,int]=None) -> str:
    have_inflections = {'NOUN', 'VERB', 'ADJ'}
    tokenized = MosesTokenizer(lang='en').tokenize(source) # Tokenize the sentence
    upper = False
    if tokenized[0][0].isupper():
        upper = True
        tokenized[0]= tokenized[0].lower()
    
    pos_tagged = nltk.pos_tag(tokenized,tagset='universal') # POS tag words in sentence
    
    for i, word in enumerate(tokenized):
        lemmas = lemminflect.getAllLemmas(word)
        # Only operate on content words (nouns/verbs/adjectives)
        if lemmas and pos_tagged[i][1] in have_inflections and pos_tagged[i][1] in lemmas:
            lemma = lemmas[pos_tagged[i][1]][0]
            inflections = (i, [(tag, infl) 
                               for tag, tup in 
                               lemminflect.getAllInflections(lemma, upos=pos_tagged[i][1]).items() 
                               for infl in tup])
            if inflections[1]:
                # Use inflection distribution for weighted random sampling if specified
                # Otherwise unweighted
                if inflection_counts:
                    counts = [inflection_counts[tag] for tag, infl in inflections[1]]
                    inflection = random.choices(inflections[1], weights=counts)[0][1]
                else:
                    inflection = random.choices(inflections[1])[0][1]
                tokenized[i] = inflection
    if upper:
        tokenized[0] = tokenized[0].title()
    return MosesDetokenizer(lang='en').detokenize(tokenized)