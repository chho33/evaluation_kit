## Pre-Request (execute first to download and install dependencies):
- python version: 3.6 up
- execute `pip install -r requirements.txt`

- Please download the following files and put them into correct directory manually:
> [coh1 model](https://drive.google.com/open?id=1FkVBItYo4ra9B-yF76o3h3WCQ9lLSs7p) 
```
mv coh1.tar.gz evaluation_kit/saved_models/coh1/;
cd evaluation_kit/saved_models/coh1;
tar zxf coh1.tar.gz --strip 1; rm coh1.tar.gz;
```
> [coh2 model](https://drive.google.com/open?id=1rJUDPJ8nng-vKUNuaSbSGlz1Ye05qOj4) 
```
mv coh2.tar.gz evaluation_kit/saved_models/coh2/;
cd evaluation_kit/saved_models/coh2;
tar zxf coh2.tar.gz --strip 1; rm coh2.tar.gz;
```
> [LM_model](https://drive.google.com/open?id=16vZJvf5_NqFabcKITOjyVYeAKfryb6Ei)
```
mv LM.tar.gz evaluation_kit/saved_models/LM/;
cd evaluation_kit/saved_models/LM;
tar zxf LM.tar.gz --strip 1; rm LM.tar.gz;
```
> [sentiment model](https://drive.google.com/open?id=112GPe7_tIoqKQwcgiXBgh6FeFPn7-ZK8) 
```
mv Model07.tar.gz evaluation_kit/saved_models/entiment_analysis/;
cd evaluation_kit/saved_models/sentiment_analysis;
tar zxf Model07.tar.gz; rm Model07.tar.gz;
```
> [fasttext_vectors](https://drive.google.com/open?id=1p3ZpcBeZcpIjMmbx3aD0QWEks-TtrPSP)
```
mv fasttext.npy.zip evaluation_kit/data/;
cd evaluation_kit/data;
unzip fasttext.npy.zip;
```
- execute `./make_dirs.sh; ./download.sh`

## Usage:
- `./run.sh [-d=$data_path|--data_path=$data_path] [-l=$log_path|--log_path=$log_path]` (default of $data_path: ./data/test_raw.csv; default of $log_path: ./logs/\`date +%s\`.log )
- example: `./run.sh -d=data/result.csv -l=logs/result.log`
##### Reminder: the format of test_raw.csv must contain "context" and "utterance" column.
##### after ./run.sh executed, the log will dump in ./logs directory.
