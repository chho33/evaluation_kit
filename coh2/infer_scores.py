from train import run 
from flags import FLAGS 
import jieba_fast as jieba
jieba.load_userdict("data/dict_fasttext.txt")

if __name__ == "__main__":
    #infer_file = "data/train_pos.csv"
    infer_file = FLAGS.inference_data_path 
    mean_prob = run([infer_file], mode="infer", jieba=jieba)
    print("coh2 score: ",mean_prob)
    if FLAGS.log_path: 
        with open(FLAGS.log_path,"a") as f:
            f.write("coh2: %s\n"%mean_prob)
