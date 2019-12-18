import os
import tensorflow as tf

# Download captions&image from internet and uzip..
# If annotations&image exit, set path to /annotations/captions_train2014.json/&/train2014/

annotation_zip=tf.keras.utils.get_file('captions.zip',
                                       cache_subdir=os.path.abspath('.'),
                                       origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                       extract=True)
annotation_file=os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
name_of_zip='train2014.zip'
if not os.path.exists(os.path.abspath('.')+'/'+name_of_zip):
    image_zip=tf.keras.utils.get_file(name_of_zip,
                                         cache_subdir=os.path.abspath('.'),
                                         origin='http://images.cocodataset.org/zips/train2014.zip',
                                         extract=True)
    PATH = os.path.dirname(image_zip)+'/train2014/'
else:
    PATH = os.path.abspath(".")+'/train2014/'
