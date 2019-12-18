from Download import annotation_file,PATH
import json
from sklearn.utils import shuffle

with open(annotation_file,'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start>' + annot['caption'] + '<end>'
    image_id = annot['image_id']
    full_coco_image_path=PATH+'COCO_train2014_'+'%012d.jpg'%(image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)
# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector=shuffle(all_captions,
                                        all_img_name_vector,
                                        )

#Select the first 30000 captions from the shuffled set
num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]