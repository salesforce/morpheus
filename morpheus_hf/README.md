# Morpheus
This repository contains code for the paper "[It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations](https://www.aclweb.org/anthology/2020.acl-main.263)" (to be presented at ACL 2020).

Authors: [Samson Tan](https://samsontmr.github.io), [Shafiq Joty](https://raihanjoty.github.io), [Min-Yen Kan](https://comp.nus.edu.sg/~kanmy), and [Richard Socher](https://socher.org)


# Supported Tasks
* NLI (MNLI)
* Question Answering (SQuAD 1.1, SQuAD 2)
* Sequence to Sequence (Summarization)
    * Easily extendable to Machine Translation with [`get_bleu` function](https://github.com/salesforce/morpheus/blob/fcb40ffe855d3363b2fa78f8fbf025dbfb9be1fa/morpheus/morpheus_nmt.py#L82)


# Supported Models
HuggingFace `transformers` models!


# Installation
```
git clone https://github.com/salesforce/morpheus.git
cd morpheus/morpheus_hf
pip install -r requirements.txt
pip install .
```


# Usage
Example:
```
from morpheus import MorpheusHuggingfaceNLI, MorpheusHuggingfaceQA

test_morph_nli = MorpheusHuggingfaceNLI('textattack/bert-base-uncased-MNLI')
premise = 'However, in the off-field (sentimental) tournament, the Falcons and Jets have more appealing story lines.'.lower()
hypothesis = 'The Jets and Falcons have boring stories.'.lower()
label = 'contradiction'

print(test_morph_nli.morph(premise, hypothesis, label))
```

# Citation
```
@inproceedings{tan-etal-2020-morphin,
    title = "It{'}s Morphin{'} Time! {C}ombating Linguistic Discrimination with Inflectional Perturbations",
    author = "Tan, Samson  and
      Joty, Shafiq  and
      Kan, Min-Yen  and
      Socher, Richard",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.263",
    pages = "2920--2935",
}
```
