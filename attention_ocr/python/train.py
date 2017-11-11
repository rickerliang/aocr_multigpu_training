# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to train the Attention OCR model.

A simple usage example:
python train.py
"""
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from tensorflow.contrib.tfprof import model_analyzer
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

import data_provider
import common_flags

#tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('num_clones', 1,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then'
                     ' the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'The frequency with which summaries are saved, in '
                     'seconds.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'Frequency in seconds of saving the model.')

flags.DEFINE_integer('max_number_of_steps', int(1e10),
                     'The maximum number of gradient steps.')

flags.DEFINE_string('checkpoint_inception', '',
                    'Checkpoint to recover inception weights from.')

flags.DEFINE_float('clip_gradient_norm', 2.0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')

flags.DEFINE_bool('sync_replicas', False,
                  'If True will synchronize replicas during training.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of gradients updates before updating params.')

flags.DEFINE_integer('total_num_replicas', 1,
                     'Total number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_boolean('reset_train_dir', False,
                     'If true will delete all files in the train_log_dir')

flags.DEFINE_boolean('show_graph_stats', False,
                     'Output model size stats to stderr.')
# yapf: enable

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate',
    'optimizer',
    'momentum',
    'use_augment_input',
])


def get_training_hparams():
  return TrainingHParams(
      learning_rate=FLAGS.learning_rate,
      optimizer=FLAGS.optimizer,
      momentum=FLAGS.momentum,
      use_augment_input=FLAGS.use_augment_input)


def create_optimizer(hparams):
  """Creates optimized based on the specified flags."""
  if hparams.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        hparams.learning_rate, momentum=hparams.momentum)
  elif hparams.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        hparams.learning_rate, momentum=hparams.momentum)
  return optimizer


def train(loss, init_fn, hparams):
  """Wraps slim.learning.train to run a training loop.

  Args:
    loss: a loss tensor
    init_fn: A callable to be executed after all other initialization is done.
    hparams: a model hyper parameters
  """
  with tf.device("/cpu:0"):
    global_step = slim.get_or_create_global_step()
  
  with tf.device("/cpu:0"):
    optimizer = create_optimizer(hparams)

  if FLAGS.sync_replicas:
    replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
    optimizer = tf.LegacySyncReplicasOptimizer(
        opt=optimizer,
        replicas_to_aggregate=FLAGS.replicas_to_aggregate,
        replica_id=replica_id,
        total_num_replicas=FLAGS.total_num_replicas)
    sync_optimizer = optimizer
    startup_delay_steps = 0
  else:
    startup_delay_steps = 0
    sync_optimizer = None

  #train_op = slim.learning.create_train_op(
  #    loss,
  #    optimizer,
  #    summarize_gradients=True,
  #    clip_gradient_norm=FLAGS.clip_gradient_norm)
  grad = optimizer.compute_gradients(loss)
  clipped_grad = tf.contrib.training.clip_gradient_norms(grad, FLAGS.clip_gradient_norm)
  update = optimizer.apply_gradients(clipped_grad, global_step=global_step)
  with tf.control_dependencies([update]):
    train_op = tf.identity(loss, name='train_op')

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.device("/cpu:0"):
    slim.learning.train(
      train_op=train_op,
      logdir=FLAGS.train_log_dir,
      graph=loss.graph,
      master=FLAGS.master,
      is_chief=(FLAGS.task == 0),
      number_of_steps=FLAGS.max_number_of_steps,
      save_summaries_secs=FLAGS.save_summaries_secs,
      save_interval_secs=FLAGS.save_interval_secs,
      startup_delay_steps=startup_delay_steps,
      sync_optimizer=sync_optimizer,
      init_fn=init_fn,
      session_config=session_config)
      
def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.
  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
  Returns:
    The list of created summaries.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(
          tf.summary.histogram(var.op.name + '_gradient', grad_values))
      summaries.append(
          tf.summary.scalar(var.op.name + '_gradient_norm',
                         clip_ops.global_norm([grad_values])))
    else:
      logging.info('Var %s has no gradient', var.op.name)

  return summaries

def _sum_clones_gradients(clone_grads):
  """Calculate the sum gradient for each shared variable across all clones.

  This function assumes that the clone_grads has been scaled appropriately by
  1 / num_clones.

  Args:
    clone_grads: A List of List of tuples (gradient, variable), one list per
    `Clone`.

  Returns:
     List of tuples of (gradient, variable) where the gradient has been summed
     across all clones.
  """
  sum_grads = []
  for grad_and_vars in zip(*clone_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      sum_grads.append((sum_grad, var))
  return sum_grads
  
def train_multigpu(losses, init_fn, hparams):
  """Wraps slim.learning.train to run a training loop.

  Args:
    loss: a loss tensor
    init_fn: A callable to be executed after all other initialization is done.
    hparams: a model hyper parameters
  """
  with tf.device("/cpu:0"):
    global_step = slim.create_global_step()
  
  with tf.device("/cpu:0"):
    optimizer = create_optimizer(hparams)

  if FLAGS.sync_replicas:
    replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
    optimizer = tf.LegacySyncReplicasOptimizer(
        opt=optimizer,
        replicas_to_aggregate=FLAGS.replicas_to_aggregate,
        replica_id=replica_id,
        total_num_replicas=FLAGS.total_num_replicas)
    sync_optimizer = optimizer
    startup_delay_steps = 0
  else:
    startup_delay_steps = 0
    sync_optimizer = None

  #train_op = slim.learning.create_train_op(
  #    loss,
  #    optimizer,
  #    summarize_gradients=True,
  #    clip_gradient_norm=FLAGS.clip_gradient_norm)
  #with tf.device("/cpu:0"):
  #  tf.summary.scalar('TotalLoss_all', total_loss)
  #  grad = optimizer.compute_gradients(total_loss)
  #with tf.device("/cpu:0"):
  #  with ops.name_scope('summarize_grads'):
  #    add_gradients_summaries(grad)
  #  clipped_grad = tf.contrib.training.clip_gradient_norms(grad, FLAGS.clip_gradient_norm)
  #  update = optimizer.apply_gradients(clipped_grad, global_step=global_step)
  #with tf.control_dependencies([update]):
  #  train_op = tf.identity(total_loss, name='train_op')
  
  # Gather update_ops from the first clone. These contain, for example,
  # the updates for the batch_norm variables created by network_fn.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "clone_0")
  
  grads = []
  total_loss = []
  for loss, i in losses:
    with tf.device("/gpu:{0}".format(i)):
      scaled_loss = tf.div(loss, 1.0 * FLAGS.num_clones)
      if i == 0:
        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        scaled_loss = scaled_loss + regularization_loss
      total_loss.append(scaled_loss)
      grad = optimizer.compute_gradients(scaled_loss)
      #if i == 0:
      #  with tf.device("/cpu:0"):
      #    with ops.name_scope("summarize_grads_{0}".format(i)):
      #      add_gradients_summaries(grad)
      grads.append(grad)
  total_loss = tf.add_n(total_loss)
  with tf.device("/cpu:0"):
    tf.summary.scalar('Total_Loss', total_loss)
  sum_grad = _sum_clones_gradients(grads)
  clipped_grad = tf.contrib.training.clip_gradient_norms(sum_grad, FLAGS.clip_gradient_norm)
  update = optimizer.apply_gradients(clipped_grad, global_step=global_step)
  update_ops.append(update)
  
  with tf.control_dependencies([tf.group(*update_ops)]):
    train_op = tf.identity(total_loss, name='train_op')
  
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  #session_config.log_device_placement = True

  with tf.device("/cpu:0"):
    slim.learning.train(
      train_op=train_op,
      logdir=FLAGS.train_log_dir,
      graph=total_loss.graph,
      master=FLAGS.master,
      is_chief=(FLAGS.task == 0),
      number_of_steps=FLAGS.max_number_of_steps,
      save_summaries_secs=FLAGS.save_summaries_secs,
      trace_every_n_steps=1000,
      save_interval_secs=FLAGS.save_interval_secs,
      startup_delay_steps=startup_delay_steps,
      sync_optimizer=sync_optimizer,
      init_fn=init_fn,
      session_config=session_config)


def prepare_training_dir():
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    logging.info('Create a new training directory %s', FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)
  else:
    if FLAGS.reset_train_dir:
      logging.info('Reset the training directory %s', FLAGS.train_log_dir)
      tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
      tf.gfile.MakeDirs(FLAGS.train_log_dir)
    else:
      logging.info('Use already existing training directory %s',
                   FLAGS.train_log_dir)


def calculate_graph_metrics():
  param_stats = model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  return param_stats.total_parameters

InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'images_orig', 'labels', 'labels_one_hot'])

def main(_):
  prepare_training_dir()

  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(dataset.num_char_classes,
                                    dataset.max_sequence_length,
                                    dataset.num_of_views, dataset.null_code)
  hparams = get_training_hparams()

  # If ps_tasks is zero, the local device is used. When using multiple
  # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
  # across the different devices.
  #device_setter = tf.train.replica_device_setter(
  #    FLAGS.ps_tasks, merge_devices=True)
  with tf.device("/cpu:0"):
    provider = data_provider.get_data(
        dataset,
        FLAGS.batch_size,
        augment=hparams.use_augment_input,
        central_crop_size=common_flags.get_crop_size())
    batch_queue = slim.prefetch_queue.prefetch_queue(
      [provider.images, provider.images_orig, provider.labels, provider.labels_one_hot], capacity=2 * FLAGS.num_clones)

  losses = []
  for i in xrange(FLAGS.num_clones):
    with tf.name_scope("clone_{0}".format(i)):
      with tf.device("/gpu:{0}".format(i)):
        #if i == 1:
        #  continue
        images, images_orig, labels, labels_one_hot = batch_queue.dequeue()
        if i == 0:
          endpoints = model.create_base(images, labels_one_hot)
        else:
          endpoints = model.create_base(images, labels_one_hot, reuse=True)
        init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint,
                                                  FLAGS.checkpoint_inception)
        if FLAGS.show_graph_stats:
          logging.info('Total number of weights in the graph: %s',
                       calculate_graph_metrics())
      
        data = InputEndpoints(
          images=images,
          images_orig=images_orig,
          labels=labels,
          labels_one_hot=labels_one_hot)
          
        total_loss, single_model_loss = model.create_loss(data, endpoints)
        losses.append((single_model_loss, i))
        with tf.device("/cpu:0"):
          tf.summary.scalar('model_loss'.format(i), single_model_loss)
          model.create_summaries_multigpu(data, endpoints, dataset.charset, i, is_training=True)
  train_multigpu(losses, init_fn, hparams)


if __name__ == '__main__':
  app.run()
