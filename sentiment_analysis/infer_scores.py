from keras.models import load_model
import pandas as pd
import numpy as np
from utils import * 

def inference(infer_data_path,model_dir,cut_mode="char"):
    df = pd.read_csv(infer_data_path)
    x_test = df.utterance.values
    x_test_word_ids = texts_to_sequences(x_test)
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_LEN)
    x_test_padded_seqs_split = get_split_list(x_test_padded_seqs,SPLIT_DIMS)
    
    model= load_model(model_dir)          
    # scores: probability of [0,1] 
    scores = model.predict(np.array(x_test_padded_seqs_split),batch_size=batch_size)
    #scores = cal_scores(scores)
    return np.mean(scores)

if __name__ == '__main__':
    if args.f: infer_data_path = args.f
    if args.model_dir: model_dir = args.model_dir
    if args.cut:
      cut_mode = args.cut
    assert infer_data_path

    mean_score = inference(infer_data_path,model_dir,cut_mode)
    print("sentiment mean_score: ",mean_score)
    if args.l:
        with open(args.l,"a") as f:
            f.write("sent: %s\n"%mean_score)
