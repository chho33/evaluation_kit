## Pre Request (execute first to download and install dependencies):
- python version: 3.6 up
- execute `pip install -r requirements.txt`
- install fasttext package for python (as illustrated below)
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```
- there are 3 files exceed 100MB that googledrive won't allow to download directly; Please download these three files and put them into correct directory manually:
> [coh1 model](https://drive.google.com/open?id=1FkVBItYo4ra9B-yF76o3h3WCQ9lLSs7p) 
```
mv coh1.tar.gz evaluation_kit/saved_model/coh1;
cd evaluation_kit/saved_model/coh1;
tar zxf coh1.tar.gz --strip 1;
```
> [LM_model](https://drive.google.com/open?id=16vZJvf5_NqFabcKITOjyVYeAKfryb6Ei)
```
mv LM.tar.gz evaluation_kit/saved_model/LM;
cd evaluation_kit/saved_model/LM;
tar zxf LM.tar.gz --strip 1;
```
> [fasttext_vectors](https://drive.google.com/open?id=1p3ZpcBeZcpIjMmbx3aD0QWEks-TtrPSP)
```
mv LM.tar.gz evaluation_kit/data;
cd evaluation_kit/data;
unzip fasttext.npy.zip;
```
- execute `./download`

## Usage:
- `./run.sh [$filename]` (if filename is not assigned, ./data/test_raw.csv will be the default file to be used.)
##### Reminder: the format of test_raw.csv must contain at least "utterance" column.
##### after ./run.sh executed, the log will dump in ./logs directory.
