from morpheus_qa import MorpheusBertQA #, MorpheusBiteBertQA
import json, os
import argparse
from tqdm import tqdm

parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file.")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
#parser.add_argument("--bite", action='store_true', required=False, help="Run on BITE-equipped model.")
args = parser.parse_args()


dataset = args.data.split('/')[-1].split('.')[0]
data = json.load(open(args.data))
total_model_queries = 0

'''
if args.bite:
    morpheus = MorpheusBiteBertQA(args.model, args.model, squad2=True)
else:
    morpheus = MorpheusBertQA(args.model, squad2=True)
'''
morpheus = MorpheusBertQA(args.model, squad2=True)

for i, article in enumerate(tqdm(data['data'], desc='article', leave=False)):
    para = 0
    num_paras = len(article['paragraphs'])
    for j, paragraph in enumerate(tqdm(article['paragraphs'], desc='paragraph', leave=False)):
        context = paragraph['context']
        for k, qa in enumerate(tqdm(paragraph['qas'], desc='question', leave=False)):
            try:
                data['data'][i]['paragraphs'][j]['qas'][k]['question'], num_queries = morpheus.morph(qa, context, constrain_pos=True, conservative=True)
            except:
                print(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                print(context)
            total_model_queries += num_queries
data['total_model_queries'] = total_model_queries

with open(os.path.join(args.output_dir, dataset + '_' + args.model + '.json'), 'w') as f:
    json.dump(data, f, indent=4)
