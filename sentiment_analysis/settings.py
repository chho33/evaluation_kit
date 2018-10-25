import os
import argparse
dirname = os.path.dirname(os.path.abspath(__file__)) 
SEED = 112
VOCAB_SIZE = 10000
BATCH_SIZE = 32
UNIT_SIZE = 256
MAX_LENGTH = 40
CHECK_STEP = 1000.
infer_file_path=os.path.join(dirname,"corpus/test_raw.csv")
file_path=os.path.join(dirname,"corpus/SAD.csv")
token_path=os.path.join(dirname,"corpus/SAD.csv.token")
mapping_path=os.path.join(dirname,"corpus/mapping")
model_dir = "save/"
batch_size = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--inference_data_path", default=None, dest="f" ,help="file path")
parser.add_argument("--mapping_path", default=None, dest="m" ,help="mapping path")
parser.add_argument("--log_path", dest="l" ,help="mean score log path")
parser.add_argument("--model_dir", help="trained model directory")
args = parser.parse_args()
