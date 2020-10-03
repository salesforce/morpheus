from .morpheus_base import MorpheusBase
from abc import abstractmethod
import torch
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag.mapping import map_tag
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from sacremoses import MosesTokenizer, MosesDetokenizer

'''
Implements `morph` and `search` methods for sequence to sequence tasks. Still an abstract class since
some key methods will vary depending on the target task and model.
'''
class MorpheusSeq2Seq(MorpheusBase):
    def __init__(self):
        self.tagger = PerceptronTagger()


    @abstractmethod
    def get_score(self, source, reference):
        pass


    def morph(self, source, reference, constrain_pos=True):
        orig_tokenized = MosesTokenizer(lang='en').tokenize(source)
        pos_tagged = [(token, map_tag("en-ptb", 'universal', tag))
                      for (token, tag) in self.tagger.tag(orig_tokenized)]
        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in pos_tagged]

        token_inflections = self.get_inflections(orig_tokenized, pos_tagged, constrain_pos)

        original_score, orig_predicted = self.get_score(source, reference)

        forward_perturbed, forward_score, \
        forward_predicted, num_queries_forward = self.search_seq2seq(token_inflections,
                                                                     orig_tokenized,
                                                                     source,
                                                                     original_score,
                                                                     reference)
        if forward_score == original_score:
            forward_predicted = orig_predicted

        if forward_score == 0:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries_forward + 1

        backward_perturbed, backward_score, \
        backward_predicted, num_queries_backward = self.search_seq2seq(token_inflections,
                                                                       orig_tokenized,
                                                                       source,
                                                                       original_score,
                                                                       reference,
                                                                       backward=True)
        if backward_score == original_score:
            backward_predicted = orig_predicted
        num_queries = 1 + num_queries_forward + num_queries_backward
        if forward_score < backward_score:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries
        else:
            return MosesDetokenizer(lang='en').detokenize(backward_perturbed), backward_predicted, num_queries

    def search_seq2seq(self, token_inflections, orig_tokenized, original,
                       original_score, reference, backward=False):
        perturbed_tokenized = orig_tokenized.copy()

        max_score = original_score
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
                curr_score, predicted = self.get_score(perturbed, reference)
                num_queries += 1
                if curr_score < max_score:
                    max_score = curr_score
                    max_infl = infl
                    max_predicted = predicted
            perturbed_tokenized[curr_token[0]] = max_infl
        return perturbed_tokenized, max_score, max_predicted, num_queries


class MorpheusSummarization(MorpheusSeq2Seq):
    def __init__(self, rouge_type='rougeL'):
        assert(rouge_type == 'rouge1' or rouge_type == 'rouge2' or rouge_type == 'rougeL')
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        self.rouge_type = rouge_type
        super().__init__()

    def get_score(self, source, reference):
        predicted = self.model_predict(source)
        return self.scorer.score(predicted, reference)[self.rouge_type].fmeasure, predicted


    @abstractmethod
    def model_predict(self, source, **kwargs):
        pass


class MorpheusHuggingfaceSummarization(MorpheusSummarization):
    def __init__(self, model_path, rouge_type='rougeL', max_input_tokens=1024, use_cuda=True):
        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path,config=config)
        self.model.eval()
        self.model.to(self.device)
        self.max_input_tokens = max_input_tokens
        super().__init__()

    def model_predict(self, source):
        tokenized = self.tokenizer.encode(source, max_length=self.max_input_tokens, return_tensors='pt')
        generated = self.model.generate(tokenized)
        print(generated)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
