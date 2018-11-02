## Pre Request (execute first to download and install dependencies):
- python version: 3.6 up
- pip install -r requirements.txt
- install fasttext package for python
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```
- ./download

## Usage:
- ./run.sh [$filename] (if not assign filename, use ./data/test_raw.csv as default)
##### Reminder: the format of test_raw.csv must contain at least "utterance" column.
##### after ./run.sh executed, the log will dump in ./logs directory.
