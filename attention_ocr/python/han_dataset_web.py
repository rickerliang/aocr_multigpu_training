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
  
  def predict(self, image_file):
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

g_predictor = Predictor()
g_predictor.initial()  

class HttpHandler(BaseHTTPRequestHandler):
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()
  
  def do_GET(self):
    if self.path == "/":
      self._set_headers()
      self.wfile.write('<! DOCTYPE html><html><head><title>Upload Image</title></head>\
      <body><form action="/aocr" method="post" enctype="multipart/form-data">\
      <input type="file" name="fileToUpload" id="fileToUpload" accept="image/*">\
      <input type="submit" value="Upload Image" name="submit"></form></body>')
  
  def do_HEAD(self):
    self._set_headers()
      
  def do_POST(self):
    # Doesn't do anything with posted data
    if self.path == "/aocr":
      form = cgi.FieldStorage(
              fp=self.rfile,
              headers=self.headers,
              environ={'REQUEST_METHOD':'POST',
                       'CONTENT_TYPE':self.headers['Content-Type'],
                       })
      file_name = form["fileToUpload"].filename
      out = g_predictor.predict(form["fileToUpload"].file)
      self._set_headers()
      out_html = "<html><head><meta charset='UTF-8'></head><body><h1>{0}</h1></body></html>".format(out)
      print("output html")
      print(out_html)
      self.wfile.write(out_html)
        
def run(server_class=HTTPServer, handler_class=HttpHandler, port=48080):
  server_address = ('', port)
  httpd = server_class(server_address, handler_class)
  print 'Starting httpd...'
  httpd.serve_forever()
  
if __name__ == "__main__":
  from sys import argv
  if len(argv) == 2:
    run(port=int(argv[1]))
  else:
    run()