import os
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
