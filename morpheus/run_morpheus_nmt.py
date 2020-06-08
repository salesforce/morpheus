from tqdm import tqdm
import argparse
from morpheus_nmt import MorpheusFairseqTransformerNMT #, MorpheusBiteFairseqTransformerNMT


parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file excluding language id.")
parser.add_argument("--src", default='en', type=str, required=False, help="Source language")
parser.add_argument("--tgt", default='de', type=str, required=False, help="Target language")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--data_len", "-l", type=int, required=False, help="Number of examples for ETA prediction.")
#parser.add_argument("--bite", action='store_true', required=False, help="Run on BITE-equipped model.")
args = parser.parse_args()

src_file = args.data + '.' + args.src
tgt_file = args.data + '.' + args.tgt
model_dir = '/'.join(args.model.split('/')[:-1])
model_file = args.model.split('/')[-1]
out_file = args.data + '.adv.' + args.src + '-' + args.tgt + '.' + model_file.split('.')[0]
#if args.bite:
#    out_file += '.bite'

print('Writing to ' + out_file)

'''
if args.bite:
    morpheus = MorpheusBiteFairseqTransformerNMT(model_dir, model_file)
else:
    morpheus = MorpheusFairseqTransformerNMT(model_dir, model_file)
'''
morpheus = MorpheusFairseqTransformerNMT(model_dir, model_file)

with open(src_file, 'r') as src_stream, open(tgt_file, 'r') as tgt_stream, open(out_file+'.'+args.src, 'w') as out_stream, open(out_file+'.pred', 'w') as pred_stream:
        for src, ref in tqdm(zip(src_stream, tgt_stream), total=args.data_len):
            adv, pred, _ = morpheus.morph(src, ref)
            out_stream.write(adv+'\n')
            pred_stream.write(pred+'\n')
