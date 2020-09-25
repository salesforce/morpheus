# Morpheus
This repository contains code for the paper "[It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations](https://www.aclweb.org/anthology/2020.acl-main.263)" (to be presented at ACL 2020).

Authors: [Samson Tan](https://samsontmr.github.io), [Shafiq Joty](https://raihanjoty.github.io), [Min-Yen Kan](https://comp.nus.edu.sg/~kanmy), and [Richard Socher](https://socher.org)


# Installation
```
pip install -r requirements.txt
cd morpheus
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
