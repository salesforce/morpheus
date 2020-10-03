# Morpheus
This repository contains code for the paper "[It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations](https://www.aclweb.org/anthology/2020.acl-main.263)" (to be presented at ACL 2020).

Authors: [Samson Tan](https://samsontmr.github.io), [Shafiq Joty](https://raihanjoty.github.io), [Min-Yen Kan](https://comp.nus.edu.sg/~kanmy), and [Richard Socher](https://socher.org)


**UPDATE:** pip installable library [here](https://github.com/salesforce/morpheus/tree/master/morpheus_hf)!


# Usage
To generate adversarial examples for one of the implemented models, run the corresponding `run_morpheus_*` script.

## Custom tasks, datasets, or models
Morpheus can be easily implemented for a custom task, dataset, or model by following the structure of existing classes:
`MorpheusBase` implements the methods common to all Morpheus implementations; `Morpheus<Task>` implements methods common to a specific task/dataset, `Morpheus<Model><Task>` implements methods specific to a particular model (usually the `init` and `morph` methods).

## Generating adversarial training data
Use `random_inflect/random_inflect.py` to generate adversarial training data. You will need to pass in a dictionary of inflection counts for it to work in the weighted sampling mode, otherwise a uniform distribution will be used. The dictionary should be in the form 

```
{
  "inflection tag": int,
}
```
E.g.,
```
{
  "VB": 150,
  "VBD": 100,
  ...
}
```
## Adversarially fine-tuned models
* [Transformer-big for WMT'14 English-French](https://storage.googleapis.com/sfr-samson-data-research/adv_ft_nmt_enfr_transformer-big.tar.gz): Compatible with `fairseq`
* [BERT-base for MNLI](https://storage.googleapis.com/sfr-samson-data-research/adv_ft_mnli_BERT-base.tar.gz): Compatible with `transformers`
* [BERT-base for SQuAD 2](https://storage.googleapis.com/sfr-samson-data-research/adv_ft_squad2_BERT-base.tar.gz): Compatible with `transformers`


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
