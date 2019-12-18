import pickle
import json
import os

path = os.path.abspath('.')+"/annotations/captions_train2014.json"
with open(path, 'r') as f:
    annotations = json.load(f)
print(annotations)
'''
for annot in annotations['annotations']:
    caption = '<start>' + annot['caption'] + '<end>'
    print(caption)
'''