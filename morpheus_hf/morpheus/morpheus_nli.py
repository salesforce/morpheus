from .morpheus_base import MorpheusBase
from abc import abstractmethod
import torch
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag.mapping import map_tag
from torch.nn import CrossEntropyLoss
from sacremoses import MosesTokenizer, MosesDetokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BartTokenizer, BartTokenizerFast, RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
from transformers import glue_processors as processors

'''
Implements `morph` and `search` methods for the MNLI task. Still an abstract class since
some key methods will vary depending on the target model.
'''
class MorpheusNLI(MorpheusBase):
    def __init__(self):
        self.tagger = PerceptronTagger()

    def morph(self, premise, hypothesis, label_text, constrain_pos=True, conservative=False):
        assert label_text in self.labels
        label = self.labels.index(label_text)
        orig_prem_tokenized = MosesTokenizer(lang='en').tokenize(premise)
        orig_hypo_tokenized = MosesTokenizer(lang='en').tokenize(hypothesis)

        prem_pos_tagged = [(token, map_tag("en-ptb", 'universal', tag))
                           for (token, tag) in self.tagger.tag(orig_prem_tokenized)]
        hypo_pos_tagged = [(token, map_tag("en-ptb", 'universal', tag))
                           for (token, tag) in self.tagger.tag(orig_hypo_tokenized)]
        prem_pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in prem_pos_tagged]
        hypo_pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in hypo_pos_tagged]

        prem_token_inflections = super().get_inflections(orig_prem_tokenized, prem_pos_tagged, constrain_pos)
        hypo_token_inflections = super().get_inflections(orig_hypo_tokenized, hypo_pos_tagged, constrain_pos)

        original_loss, init_predicted = self.get_loss(premise, hypothesis, label)
        if init_predicted != label:
            return premise, hypothesis, self.labels[init_predicted], 1

        forward_prem_perturbed, \
        forward_hypo_perturbed, \
        forward_loss, forward_predicted, \
        num_queries_forward = self.search_nli(prem_token_inflections,
                                              hypo_token_inflections,
                                              orig_prem_tokenized,
                                              orig_hypo_tokenized,
                                              original_loss,
                                              label,
                                              conservative)

        if conservative and forward_predicted != label:
            return forward_prem_perturbed, forward_hypo_perturbed, self.labels[forward_predicted], num_queries_forward + 1

        backward_prem_perturbed, \
        backward_hypo_perturbed, \
        backward_loss, backward_predicted, \
        num_queries_backward = self.search_nli(prem_token_inflections,
                                               hypo_token_inflections,
                                               orig_prem_tokenized,
                                               orig_hypo_tokenized,
                                               original_loss,
                                               label,
                                               conservative)

        num_queries = 1 + num_queries_forward + num_queries_backward
        if forward_loss > backward_loss:
            return forward_prem_perturbed, forward_hypo_perturbed, self.labels[forward_predicted], num_queries
        else:
            return backward_prem_perturbed, backward_hypo_perturbed, self.labels[backward_predicted], num_queries

    def search_nli(self, prem_token_inflections, hypo_token_inflections, orig_prem_tokenized,
                   orig_hypo_tokenized, original_loss, label, conservative=True, backward=False):
        perturbed_prem_tokenized = orig_prem_tokenized.copy()
        perturbed_hypo_tokenized = orig_hypo_tokenized.copy()


        max_loss = original_loss
        num_queries = 0
        max_predicted = label

        if backward:
            prem_token_inflections = reversed(prem_token_inflections)
            hypo_token_inflections = reversed(hypo_token_inflections)

        detokenizer = MosesDetokenizer(lang='en')
        premise = detokenizer.detokenize(perturbed_prem_tokenized)

        for curr_token in hypo_token_inflections:
            max_infl = orig_hypo_tokenized[curr_token[0]]
            for infl in curr_token[1]:
                perturbed_hypo_tokenized[curr_token[0]] = infl
                perturbed = detokenizer.detokenize(perturbed_hypo_tokenized)
                loss, predicted = self.get_loss(premise, perturbed, label)
                num_queries += 1
                if loss > max_loss:
                    max_loss = loss
                    max_infl = infl
                    max_predicted = predicted
                if conservative and predicted != label:
                    break
            perturbed_hypo_tokenized[curr_token[0]] = max_infl

        hypothesis = detokenizer.detokenize(perturbed_hypo_tokenized)

        for curr_token in prem_token_inflections:
            max_infl = orig_prem_tokenized[curr_token[0]]
            for infl in curr_token[1]:
                perturbed_prem_tokenized[curr_token[0]] = infl
                perturbed = detokenizer.detokenize(perturbed_prem_tokenized)
                loss, predicted = self.get_loss(perturbed, hypothesis, label)
                num_queries += 1
                if loss > max_loss:
                    max_loss = loss
                    max_infl = infl
                    max_predicted = predicted
                if conservative and predicted != label:
                    break
            perturbed_prem_tokenized[curr_token[0]] = max_infl

        premise = detokenizer.detokenize(perturbed_prem_tokenized)

        return premise, hypothesis, max_loss, max_predicted, num_queries

'''
Implements model-specific details. Tested on BERT/RoBERTa-based models
'''
class MorpheusHuggingfaceNLI(MorpheusNLI):
    labels = ["contradiction", "entailment", "neutral"]
    def __init__(self, model_path, use_cuda=True):
        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.__class__ in (RobertaTokenizer,
                                        RobertaTokenizerFast,
                                        XLMRobertaTokenizer,
                                        BartTokenizer,
                                        BartTokenizerFast):
            # hack to handle roberta models
            self.labels[1], self.labels[2] = self.labels[2], self.labels[1]

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,config=config)

        self.model.eval()
        self.model.to(self.device)
        self.loss_fn = CrossEntropyLoss()
        super().__init__()

    def get_loss(self, premise, hypothesis, label, max_seq_len=128):
        logits, _ = self.model_predict(premise, hypothesis, max_seq_len)
        label_tensor = torch.tensor([label]).to(self.device)
        loss = self.loss_fn(logits, label_tensor)

        return loss.item(), logits.argmax().item()


    def model_predict(self, premise, hypothesis, max_seq_len=512):
        inputs = self.tokenizer.encode_plus(premise, hypothesis, add_special_tokens=True, max_length=max_seq_len)
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
        if "token_type_ids" in inputs.keys():
            token_type_ids = torch.tensor(inputs["token_type_ids"]).unsqueeze(0).to(self.device)
        else:
            # handle XLM-R
            token_type_ids = torch.tensor([0]*len(input_ids)).unsqueeze(0).to(self.device)

        outputs = self.model(input_ids,token_type_ids=token_type_ids)
        logits = outputs[0]
        return logits, self.labels[logits.argmax().item()]
