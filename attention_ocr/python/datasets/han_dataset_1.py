import fsns

DEFAULT_DATASET_DIR = '/home/lyk/machine_learning/Supervised_Learning/tensorflow_models/attention_ocr/python/han_dataset_1'

DEFAULT_CONFIG = {
    'name':
        'han_datasets_1',
    'splits': {
        'train': {
            'size': 179736,
            'pattern': 'han_dataset_train.*'
        },
        'test': {
            'size': 77024,
            'pattern': 'han_dataset_test.*'
        }
    },
    'charset_filename':
        'charset.txt',
    'image_shape': (50, 380, 3),
    'num_of_views':
        1,
    'max_sequence_length':
        18,
    'null_code':
        1977,
    'items_to_descriptions': {
        'image':
            'A [50 x 380 x 3] color image.',
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
