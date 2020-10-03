from .morpheus_base import MorpheusBase
from abc import abstractmethod
import torch
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag.mapping import map_tag
from torch.nn import CrossEntropyLoss
from sacremoses import MosesTokenizer, MosesDetokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from .squad_utils import compute_f1

'''
Implements `morph` and search methods for the SQuAD 2 task. Still an abstract class since
some key methods will vary depending on the target model.
'''
class MorpheusQA(MorpheusBase):
    def __init__(self):
        self.tagger = PerceptronTagger()

    def morph(self, question_dict, context, constrain_pos=True, conservative=False):
        original = question_dict['question']

        gold_starts = [ans['answer_start'] for ans in question_dict['answers']]
        gold_texts = [ans['text'] for ans in question_dict['answers']]
        gold_ends = [gold_starts[i]+len(text) for i, text in enumerate(gold_texts)]
        question_dict['gold_char_spans'] = list(zip(gold_starts, gold_ends))
        question_dict['gold_texts'] = gold_texts

        orig_tokenized = MosesTokenizer(lang='en').tokenize(original)

        pos_tagged = [(token, map_tag("en-ptb", 'universal', tag))
                      for (token, tag) in self.tagger.tag(orig_tokenized)]
        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in pos_tagged]

        token_inflections = super(MorpheusQA, self).get_inflections(orig_tokenized, pos_tagged, constrain_pos)

        original_loss, init_predicted = self.get_loss(original, question_dict, context)

        if self.metric_max(compute_f1, init_predicted, question_dict['gold_texts']) == 0:
            return original, init_predicted, 1

        forward_perturbed, \
        forward_loss, forward_predicted, \
        num_queries_forward = self.search_qa(token_inflections,
                                             orig_tokenized,
                                             original_loss,
                                             question_dict,
                                             context,
                                             conservative)

        if conservative and self.metric_max(compute_f1, forward_predicted, question_dict['gold_texts']) == 0:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries_forward + 1

        backward_perturbed, \
        backward_loss, backward_predicted, \
        num_queries_backward = self.search_qa(token_inflections,
                                              orig_tokenized,
                                              original_loss,
                                              question_dict,
                                              context,
                                              conservative,
                                              backward=True)


        num_queries = 1 + num_queries_forward + num_queries_backward
        if forward_loss > backward_loss:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries
        else:
            return MosesDetokenizer(lang='en').detokenize(backward_perturbed), backward_predicted, num_queries

    def search_qa(self, token_inflections, orig_tokenized, original_loss, question_dict, context, conservative=True, backward=False):
        perturbed_tokenized = orig_tokenized.copy()

        max_loss = original_loss
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
                loss, predicted = self.get_loss(perturbed, question_dict, context)
                num_queries += 1
                if loss > max_loss:
                    max_loss = loss
                    max_infl = infl
                    max_predicted = predicted
                if conservative and self.metric_max(compute_f1, predicted, question_dict['gold_texts']) == 0:
                    break
            perturbed_tokenized[curr_token[0]] = max_infl
        return perturbed_tokenized, max_loss, max_predicted, num_queries

    @staticmethod
    def get_lowest_loss(start_logits_tensor, end_logits_tensor, gold_spans):
        start_logits_tensor = start_logits_tensor.to('cpu')
        end_logits_tensor = end_logits_tensor.to('cpu')
        target_tensors = [(torch.tensor([gold_start]), torch.tensor([gold_end])) \
                              for gold_start, gold_end in gold_spans]
        loss = CrossEntropyLoss()
        losses = []
        #print(start_logits_tensor.shape, end_logits_tensor.shape)
        #print(target_tensors)
        for target_start, target_end in target_tensors:
            avg_loss = (loss(start_logits_tensor, target_start) \
                        + loss(end_logits_tensor, target_end))/2
            losses.append(avg_loss)
        return min(losses).item()

    @staticmethod
    def metric_max(metric, pred, labels):
        scores = []
        for label in labels:
            score = metric(pred, label)
            scores.append(score)
        return max(scores)

'''
Implements model-specific details. Tested on BERT/RoBERTa-based models
'''
class MorpheusHuggingfaceQA(MorpheusQA):
    def __init__(self, model_path, squad2=True, use_cuda=True):
        self.squad2=squad2

        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path,config=config)
        self.num_special_tokens = len(self.tokenizer.build_inputs_with_special_tokens([],[]))
        self.model.eval()
        self.model.to(self.device)
        super().__init__()

    def morph(self, question_dict, context, constrain_pos=True, conservative=False):
        if not self.squad2 and question_dict.get('is_impossible'):
            raise AssertionError('Not SQuAD 2 but has impossible questions')
        return super().morph(question_dict, context, constrain_pos, conservative)


    def get_loss(self, question, question_dict, context, max_seq_len=512, max_query_len=64):
        start_logits, end_logits, predicted = self.model_predict(question, context, max_seq_len, max_query_len)
        question_dict['gold_spans'] = list(set(self.get_gold_token_spans(self.tokenizer, question,
                                                                         question_dict, context,
                                                                         max_seq_len, max_query_len)))

        return super().get_lowest_loss(start_logits, end_logits, question_dict['gold_spans']), predicted

    @staticmethod
    def get_gold_token_spans(tokenizer, question, question_dict, context, max_seq_len, max_query_len):
        token_spans = []
        num_special_tokens = len(tokenizer.build_inputs_with_special_tokens([],[]))


        if question_dict.get('is_impossible'):
            question_dict['gold_texts'] = ['']
            return [(0, 0)]


        qn_ids = tokenizer.encode(question, add_special_tokens=False)
        #context_ids = tokenizer.encode(context, add_special_tokens=False)
        #print(tokenizer.convert_ids_to_tokens(tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids)))

        if len(qn_ids) > max_query_len:
            qn_ids = qn_ids[:max_query_len]

        for i, (char_span_start, char_span_end) in enumerate(question_dict['gold_char_spans']):
            gold_text = question_dict['gold_texts'][i]
            if gold_text == '.':
                # Ignore garbage annotation
                continue
            # Handle SentencePiece tokenizers
            if context[char_span_start-1] == ' ' and len(tokenizer.encode(context[:char_span_start], add_special_tokens=False)) != len(tokenizer.encode(context[:max(char_span_start-1,0)], add_special_tokens=False)):
                char_span_start -= 1
                gold_text = ' ' + gold_text

            span_start = len(qn_ids) + num_special_tokens - 1 + len(tokenizer.encode(context[:char_span_start], add_special_tokens=False))
            span_end = span_start + len(tokenizer.encode(gold_text, add_special_tokens=False))

            if span_start >= max_seq_len:
                span_start = 0
                span_end = 0
                question_dict['gold_texts'][i] = ''
            elif span_end >= max_seq_len:
                span_end = max_seq_len - 1
                print(span_end)
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                max_context_len = max_seq_len - len(qn_ids) - num_special_tokens
                context_ids = context_ids[:max_context_len]
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids))
                question_dict['gold_texts'][i] = tokenizer.convert_tokens_to_string(tokens[span_start:max_seq_len]).strip()

            token_spans.append((span_start, span_end))
        return token_spans

    def model_predict(self, question, context, max_seq_len=512, max_query_len=64):
        qn_ids = self.tokenizer.encode(question, add_special_tokens=False)
        if len(qn_ids) > max_query_len:
            qn_ids = qn_ids[:max_query_len]
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        max_context_len = max_seq_len - len(qn_ids) - self.num_special_tokens
        if len(context_ids) > max_context_len:
            context_ids = context_ids[:max_context_len]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(qn_ids, context_ids)
        with torch.no_grad():
            start_logits, end_logits = self.model(torch.tensor([input_ids]).to(self.device), token_type_ids=torch.tensor([token_type_ids]).to(self.device))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        predicted = self.tokenizer.convert_tokens_to_string(all_tokens[torch.argmax(start_logits): torch.argmax(end_logits)+1])
        if predicted == self.tokenizer.cls_token:
            predicted = ''
        return start_logits, end_logits, predicted
