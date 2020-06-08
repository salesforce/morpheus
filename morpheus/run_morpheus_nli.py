from transformers import glue_processors as processors
import csv, os
from morpheus_nli import MorpheusBertNLI #, MorpheusBiteBertNLI
from tqdm import tqdm
import argparse

def _create_output_data(examples, input_tsv_list):
    output = []
    for (i, line) in enumerate(input_tsv_list):
            output_line = line.copy()
            if i == 0:
                output_line.insert(-1, 'predicted_label')
                output.append(output_line)
                continue
            output_line[4] = '-'
            output_line[5] = '-'
            output_line[6] = '-'
            output_line[7] = '-'
            output_line[8] = examples[i-1].text_a
            output_line[9] = examples[i-1].text_b
            try:
                output_line.insert(-1, examples[i-1].adv_label)
            except AttributeError:
                output_line.insert(-1, '-')
            output.append(output_line)
    return output

def _write_tsv(output, output_file):
    with open(output_file, "w", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter="\t", quotechar=None)
        for row in output:
            writer.writerow(row)

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data directory, e.g., 'data/MNLI'.")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
#parser.add_argument("--bite", action='store_true', required=False, help="Run on BITE-equipped model.")
parser.add_argument("--mm", action='store_true', required=False, help="Use Mismatch dev data.")
args = parser.parse_args()

if args.mm:
    input_tsv = processors['mnli-mm']()._read_tsv(args.data+'/dev_mismatched.tsv')
else:
    input_tsv = processors['mnli']()._read_tsv(args.data+'/dev_matched.tsv')

'''
if args.bite:
    morpheus = MorpheusBiteBertNLI(args.model, args.model)
else:
    morpheus = MorpheusBertNLI(args.model)
'''
morpheus = MorpheusBertNLI(args.model)

if args.mm:
    examples = processors['mnli-mm']().get_dev_examples(args.data)
else:
    examples = processors['mnli']().get_dev_examples(args.data)

for example in tqdm(examples):
    prem, hypo, text_label, _ = morpheus.morph(example.text_a, example.text_b, example.label)
    if text_label != example.label:
        print('original: ', example.text_a, example.text_b, example.label)
        example.text_a = prem
        example.text_b = hypo
        example.adv_label = text_label
        print('adversarial: ', prem, hypo, text_label)

output_path = os.path.join(args.output_dir, 'mnli')
if args.mm:
    output_path += '-mm'
output_path += '_dev_' + args.model + '.tsv'
_write_tsv(_create_output_data(examples, input_tsv), output_path)
