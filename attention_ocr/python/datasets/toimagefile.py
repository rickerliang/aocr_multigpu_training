import numpy as np
import skimage.io as io
import tensorflow as tf

record_iterator = tf.python_io.tf_record_iterator('testdata/fsns/fsns-00000-of-00001')

for record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(record)
  #print example.features.feature['image/encoded'].bytes_list.value[0]
  f = open('test.png', 'wb')
  f.write(example.features.feature['image/encoded'].bytes_list.value[0])
  f.close()
  print example.features.feature['image/format'].bytes_list.value[0]
  print example.features.feature['image/width'].int64_list.value[0]
  print example.features.feature['image/orig_width'].int64_list.value[0]
  print example.features.feature['image/class'].int64_list.value
  print example.features.feature['image/unpadded_class'].int64_list.value
  print example.features.feature['image/text'].bytes_list.value[0]
  break
