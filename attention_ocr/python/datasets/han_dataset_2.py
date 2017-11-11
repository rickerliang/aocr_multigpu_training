import fsns

DEFAULT_DATASET_DIR = '/home/lyk/machine_learning/Supervised_Learning/tensorflow_models/attention_ocr/python/han_dataset_2'

DEFAULT_CONFIG = {
    'name':
        'han_datasets_2',
    'splits': {
        'train': {
            'size': 3364920 / 15,
            'pattern': 'han_dataset_train_NotoSansCJK-Light_15.tfrecords'
        },
        'test': {
            'size': 1442160 / 15,
            'pattern': 'han_dataset_test__NotoSansCJK-Light_15.tfrecords'
        }
    },
    'charset_filename':
        'charset.txt',
    'image_shape': (50, 780, 3),
    'num_of_views':
        1,
    'max_sequence_length':
        40,
    'null_code':
        3310,
    'items_to_descriptions': {
        'image':
            'A [50 x 780 x 3] color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config)
