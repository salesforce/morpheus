# Morpheus
This repository contains code for the paper "[It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations](https://arxiv.org/abs/2005.04364)" (to be presented at ACL 2020).

Authors: [Samson Tan](https://samsontmr.github.io), [Shafiq Joty](https://raihanjoty.github.io), [Min-Yen Kan](https://comp.nus.edu.sg/~kanmy), and [Richard Socher](https://socher.org)


# Usage
To generate adversarial examples for one of the implemented models, run the corresponding `run_morpheus_*` script.

## Custom tasks, datasets, or models
Morpheus can be easily implemented for a custom task, dataset, or model by following the structure of existing classes:
`MorpheusBase` implements the methods common to all Morpheus implementations; `Morpheus<Task>` implements methods common to a specific task/dataset, `Morpheus<Model><Task>` implements methods specific to a particular model (usually the `init` and `morph` methods).

## Generating adversarial training data
Coming soon.

# Citation
```
@inproceedings{morpheus20,
  author    = {Samson Tan and Shafiq Joty and Min-Yen Kan and Richard Socher},
  title     = {It’s {M}orphin’ {T}ime! {C}ombating Linguistic Discrimination with Inflectional Perturbations},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2020},
  address   = {Seattle, Washington},
  numpages  = {9},
  publisher = {Association for Computational Linguistics},
  url       = {https://arxiv.org/abs/2005.04364}
}
```
