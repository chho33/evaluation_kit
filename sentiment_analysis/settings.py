import os
import argparse
dirname = os.path.dirname(os.path.abspath(__file__)) 

parser = argparse.ArgumentParser()
parser.add_argument("--inference_data_path", default=None, dest="f" ,help="file path")
parser.add_argument("--mapping_path", default=None, dest="m", help="mapping path")
parser.add_argument("--log_path", dest="l", help="mean score log path")
parser.add_argument("--model_dir", help="trained model directory")
parser.add_argument("--model_type", default="cnn", help="[cnn|rnn_last|rnn_ave]")
parser.add_argument("--sentence_cut_mode", default="word", dest="cut", help="how to cut sentence? [word|char]")
parser.add_argument("--jieba_dict", default="dict_fasttext.txt", help="setup jieba dictionary")
parser.add_argument("--vocab_size", default=50000, help="vocabulary size")
args = parser.parse_args()
if args.cut and args.cut == "word":
    mapping_path=os.path.join(dirname,"corpus/word_mapping")
elif args.cut and args.cut == "char":
    mapping_path=os.path.join(dirname,"corpus/char_mapping")

SEED = 112
VOCAB_SIZE = args.vocab_size 
BATCH_SIZE = 32
UNIT_SIZE = 256
MAX_LENGTH = 40
CHECK_STEP = 1000.
infer_file_path=os.path.join(dirname,"corpus/test_raw.csv")
file_path=os.path.join(dirname,"corpus/SAD.csv")
token_path=os.path.join(dirname,"corpus/SAD.csv.token")
model_dir = "save/"
batch_size = 1000
cut_mode = "char"
