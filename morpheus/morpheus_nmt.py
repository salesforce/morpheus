# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
sys.path.append(".")
#from base_inflect import BITETokenizer
from morpheus_base import MorpheusBase
from abc import abstractmethod
import sacrebleu
from fairseq.models.transformer import TransformerModel
import nltk, torch
from sacremoses import MosesTokenizer, MosesDetokenizer

'''
Implements `morph` and `search` method for the MT task. Still an abstract class since 
some key methods will vary depending on the target model.
'''
class MorpheusNMT(MorpheusBase):
    @abstractmethod
    def __init__(self):
        pass
    
    def morph(self, source, reference, constrain_pos=True):
        orig_tokenized = MosesTokenizer(lang='en').tokenize(source)
        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in nltk.pos_tag(orig_tokenized,tagset='universal')]

        token_inflections = self.get_inflections(orig_tokenized, pos_tagged, constrain_pos)

        original_bleu, orig_predicted = self.get_bleu(source, reference)

        forward_perturbed, forward_bleu, forward_predicted, num_queries_forward = self.search_nmt(token_inflections,
                                                                           orig_tokenized,
                                                                           source,
                                                                           original_bleu,
                                                                           reference)
        if forward_bleu == original_bleu:
            forward_predicted = orig_predicted

        if forward_bleu == 0:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries_forward + 1
        
        backward_perturbed, backward_bleu, backward_predicted, num_queries_backward = self.search_nmt(token_inflections,
                                                                              orig_tokenized,
                                                                              source,
                                                                              original_bleu,
                                                                              reference,
                                                                              backward=True)
        if backward_bleu == original_bleu:
            backward_predicted = orig_predicted
        num_queries = 1 + num_queries_forward + num_queries_backward
        if forward_bleu < backward_bleu:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries
        else:
            return MosesDetokenizer(lang='en').detokenize(backward_perturbed), backward_predicted, num_queries
        
    def search_nmt(self, token_inflections, orig_tokenized, original, 
                   original_bleu, reference, backward=False):
        perturbed_tokenized = orig_tokenized.copy()

        max_bleu = original_bleu
        num_queries = 0
        max_predicted = ''

        if backward:
            token_inflections = reversed(token_inflections)
            
        detokenizer = MosesDetokenizer(lang='en')
        
        for curr_token in token_inflections:
            max_infl = orig_tokenized[curr_token[0]]
            for infl in curr_token[1]:
                perturbed_tokenized[curr_token[0]] = infl
                perturbed = detokenizer.detokenize(perturbed_tokenized)
                curr_bleu, predicted = self.get_bleu(perturbed, reference)
                num_queries += 1
                if curr_bleu < max_bleu:
                    max_bleu = curr_bleu
                    max_infl = infl
                    max_predicted = predicted
            perturbed_tokenized[curr_token[0]] = max_infl
        return perturbed_tokenized, max_bleu, max_predicted, num_queries
    
    def get_bleu(self, source, reference, beam=5):
        predicted = self.model.translate(source, beam)
        return sacrebleu.sentence_bleu(predicted, reference).score, predicted


'''
Implements model-specific details.
'''
class MorpheusFairseqTransformerNMT(MorpheusNMT):
    def __init__(self, model_dir, model_file, tokenizer='moses', bpe='subword_nmt', use_cuda=True):
        self.model = TransformerModel.from_pretrained(model_dir, model_file, tokenizer=tokenizer, bpe=bpe)
        if use_cuda and torch.cuda.is_available():
            self.model.cuda()

'''
class MorpheusBiteFairseqTransformerNMT(MorpheusFairseqTransformerNMT):
    def __init__(self, model_dir, model_file, tokenizer='moses', bpe='subword_nmt', use_cuda=True):
        super().__init__(model_dir, model_file, tokenizer, bpe, use_cuda)
        self.bite = BITETokenizer()
    
    def get_bleu(self, source, reference, beam=5):
        bite_source = ' '.join(self.bite.tokenize(source))
        return super().get_bleu(bite_source, reference, beam)
'''
