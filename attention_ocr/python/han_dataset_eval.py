import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from PIL import Image
import numpy as np

import data_provider
import common_flags

FLAGS = flags.FLAGS
common_flags.define()
flags.DEFINE_string('input_image', 'input00.png',
                    'Image to eval.')

def main(_):
  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(dataset.num_char_classes,
                                  dataset.max_sequence_length,
                                  dataset.num_of_views, dataset.null_code,
                                  charset=dataset.charset)
  data = data_provider.get_data(
      dataset,
      FLAGS.batch_size,
      augment=False,
      central_crop_size=common_flags.get_crop_size())


  input_image = Image.open(FLAGS.input_image).convert("RGB").resize((data.images.shape[2]
    / dataset.num_of_views, data.images.shape[1]))
  input_array = np.array(input_image).astype(np.float32)
  #input_array = np.concatenate((input_array, input_array, input_array, input_array), axis=1)
  #Image.fromarray(input_array.astype(np.uint8), "RGB").save("test_input_1.jpg")
  input_array = np.expand_dims(input_array, axis=0)
  print input_array.shape
  print input_array.dtype
  #input_image.save("test_input.jpg")
  #return
  
  placeholder_shape = (1, data.images.shape[1], data.images.shape[2], data.images.shape[3])
  print placeholder_shape
  image_placeholder = tf.placeholder(tf.float32, shape=placeholder_shape)
  endpoints = model.create_base(image_placeholder, labels_one_hot=None)
  init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint)
  with tf.Session() as sess:
    tf.tables_initializer().run()  # required by the CharsetMapper
    init_fn(sess)
    predictions = sess.run(endpoints.predicted_text,
                           feed_dict={image_placeholder: input_array})
  print("Predicted strings:")
  for line in predictions:
    print(line)
  
if __name__ == '__main__':
  app.run()  
  