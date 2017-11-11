import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import cgi
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from PIL import Image
import numpy as np
import cv2

import data_provider
import common_flags

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = flags.FLAGS
common_flags.define()


config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

class Predictor():
  def initial(self):
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
        
    self.image_height = int(data.images.shape[1])
    self.image_width = int(data.images.shape[2])
    self.image_channel = int(data.images.shape[3])
    self.num_of_view = dataset.num_of_views
    placeholder_shape = (1, self.image_height, self.image_width, self.image_channel)
    print placeholder_shape
    self.placeholder = tf.placeholder(tf.float32, shape=placeholder_shape)
    self.endpoint = model.create_base(self.placeholder, labels_one_hot=None)    
    init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint)
    
    self.sess = tf.Session(config=config)
    tf.tables_initializer().run(session=self.sess)
    init_fn(self.sess)

  def capture_text_regions(self, image):
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    blur = cv2.GaussianBlur(img2gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('out_image_threshold.png', th3)
    # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))    
    # dilate , more the iteration more the dilation
    dilated = cv2.dilate(th3, kernel, iterations=9)    
    # get contours
    image_contour, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imwrite('out_image_4.png', image_contour)
    print(len(contours))
    image_segments=[]
    for contour in contours:
      # get rectangle bounding contour
      [x, y, w, h] = cv2.boundingRect(contour)
    
      # Don't plot small false positives that aren't text
      if w < 35 and h < 35:
        continue
    
      ## draw rectangle around contour on original image
      #cv2.rectangle(img2gray, (x, y), (x + w, y + h), (255, 0, 255), 2)
      #'''
      #    #you can crop image and send to OCR  , false detected will return no text :)
      #    cropped = img_final[y :y +  h , x : x + w]
      #
      #    s = file_name + '/crop_' + str(index) + '.jpg' 
      #    cv2.imwrite(s , cropped)
      #    index = index + 1
      #
      #    '''
      # write original image with added contours to disk
      image_segments.append(img2gray[y : y + h , x : x + w])
    return image_segments
      
  def create_blank(self, width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image
      
  def image_segments_to_text(self, image_segments):
    width = self.image_width / self.num_of_view
    height = self.image_height
    text_lines=[]
    for idx, segment in enumerate(image_segments):
      #print "segment.shape ", segment.shape
      if segment.shape[1] + 0 < width and segment.shape[0] < height:
        background = self.create_blank(width, height, (255, 255, 255))
        #print "segment.shape ", segment.shape
        #print "background.shape ", background.shape
        height_begin = (height - segment.shape[0]) / 2
        #print background[height_begin:height_begin+segment.shape[0], 20:20+segment.shape[1]].shape
        three_channel = np.zeros((segment.shape[0], segment.shape[1], 3), np.uint8)
        three_channel[:,:,0] = segment
        three_channel[:,:,1] = segment
        three_channel[:,:,2] = segment
        background[height_begin:height_begin+segment.shape[0], 0:0+segment.shape[1]] = three_channel
        cv2.imwrite('/tmp/composed{0}.png'.format(idx), background)
        input_array = np.array(background).astype(np.float32)
        input_array = np.expand_dims(input_array, axis=0)
   
        predictions = self.sess.run(self.endpoint.predicted_text,
                             feed_dict={self.placeholder: input_array})
        print("Predicted strings {0}:".format(idx))
        for idx, line in enumerate(predictions):
          print line
          #print idx
          text_lines.append(line)
    return text_lines
    
  def image_file_to_text(self, image_file):
    input_image = Image.open(image_file).convert("RGB").resize((self.image_width
      / self.num_of_view, self.image_height))
    input_image.save("image_resized.png")
    input_array = np.array(input_image).astype(np.float32)
    input_array = np.expand_dims(input_array, axis=0)
   
    predictions = self.sess.run(self.endpoint.predicted_text,
                             feed_dict={self.placeholder: input_array})
    print("Predicted strings:")
    for idx, line in enumerate(predictions):
      print line
      print idx
    return line
    
  def image_file_to_text_line(self, image_file):
    input_image = np.array(Image.open(image_file).convert("RGB"))
    image_segments = self.capture_text_regions(input_image)
    self.image_segments_to_text(image_segments)
    
predictor = Predictor()
predictor.initial()
predictor.image_file_to_text_line('raw.png')