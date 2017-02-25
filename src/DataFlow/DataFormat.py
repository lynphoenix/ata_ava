import tensorflow as tf
import random
import math
import sys
import os
from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5



'''
standard data Format
{
  "source": "/disk2/image1.jpg",
  "type": "image", //image, video, text ...
  "label":{
    "classification": 1
  }
}

'''


class ImageOps(object):

    def __init__(self, image_file):
        try:
            fp = open(path)
            self.contents = fp.read()[:4]
            self.substr = string_ops.substr(contents, 0, 4)
            self.is_open = True
        except Exception as e:
            print(str(e))
            self.is_open = False

    def _is_gif():
        return self.substr == b'\x47\x49\x46\x38'

    def _is_jpeg():
        return self.substr == b'\xff\xd8\xff\xe0'

    def _is_png():
        return self.substr == b'\211PNG'

    def identify_format():
        if self.is_open:
            if _is_jpeg():
                return True, "jepg"
            elif _is_png():
                return True, "png"
            elif _is_gif():
                return True, "gif"
        else:
            return False, "Unknown"


class ImageReader(object):

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decode_image = tf.image.decode_image(self._decode_data, channels=3)

    def read_image(self, sess, image_data, image_file):
        image = self.decode_image(sess, image_data, image_file)
        return image

    def decode_image(self, sess, image_data, image_file):
        try:
            image = sess.run(self._decode_image, feed_dict={self._decode_data: image_data})
            if len(image.shape) == 3 and image.shape[2] == 3:
                return True, image
            else:
                print("image dim error :" + image_file)
                return False, image
        except Exception as e:
            print("decode_image exception: " + image_file)
            return False, None


def _recjson_image_cls(rec_log):
    try:
        recjson = eval(rec_log)
        file_name = recjson["source"]
        assert recjson["type"] == "image"
        label = recjson["label"]["classification"]
        assert type(label) == int
        return True, file_name, label
    except:
        print("_analyse_recjson_image_cls Error " + rec_log)
        return False, "RecjsonNone"


def _log_to_image_cls(rec_log):
    try:
        file_name = rec_log[:-2]
        label = int(rec_log[-1:])
        return True, file_name, label
    except:
        print("_log_to_image_cls Error " + rec_log)
        return False, None, None


def _get_imagedata(rec_log, sess, image_reader, idx):
    try:
        res, image_file, label = _log_to_image_cls(rec_log)
        if not res:
            return False, {}
        image_binary = tf.gfile.FastGFile(image_file, 'r').read()
        res, image = image_reader.read_image(sess, image_binary, image_file)
        if res:
            return True, {"image_data": image_binary, "height": image.shape[0], "width": image.shape[1], "label": label}
        else:
            return False, {}
    except Exception as e:
        print("Read Error: %s-%d" % (image_file, idx))
        return False, {}


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, data_list, dataset_dir):

    assert split_name in ['train', 'validation']
    num_per_shard = int(math.ceil(len(data_list) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(data_list))

                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(data_list), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        res, res_json = _get_imagedata(data_list[i], sess, image_reader, i)
                        if not res:
                            continue
                        example = dataset_utils.image_to_tfexample(res_json["image_data"], 'jpg', res_json["height"], res_json["width"], res_json["label"])
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _convert_list(file_list):

    out_list = []
    fp = open(file_list, "r")

    for f in fp:
        f = f.strip()
        out_list.append(f)
    fp.close()
    return out_list


def _convert_json(file_list):

    out_list = []
    fp = open(file_list, "r")

    for f in fp:
        f = f.strip()
        try:
            log = eval(f)
            out_list.append(f.eval())
        except Exception as e:
            print("log eval error " + str(e))

    fp.close()
    return out_list



def split_train_val_test(data_list, train_ratio=0.9, test_ratio=0., is_suffle=True):
    if is_suffle:
        random.shuffle(data_list)

    total_len = len(data_list)
    train_len = int(total_len * train_ratio)
    train_list = data_list[:train_len]

    if test_ratio > 0.:
        test_len = int(total_len * test_ratio)
        test_list = data_list[total_len - test_len + 1:]
        val_list = data_list[train_len+1:total_len - test_len + 1]
        return train_list, val_list, test_list
    else:
        test_len = 0
        val_list = data_list[train_len + 1 :]
        return train_list, val_list

def Std2Tf(dataset_dir, label_file, data_file, train_ratio=0.9):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(label_file, 'r').readlines()]

    # First, convert the training and validation sets.
    data_list = _convert_list(data_file)
    train_list, val_list = split_train_val_test(data_list, train_ratio)

    _convert_dataset('validation', val_list, dataset_dir)
    _convert_dataset('train', train_list, dataset_dir)

    # Finally, write the labels file:
    dataset_utils.write_label_file(unique_labels, dataset_dir)

def Std2LMDB():
    pass

def Std2MXRec():
    pass


def test(data_file):
    dataset_dir = "/disk2/data/ava_test/data"
    label_file = "/disk2/data/ava_test/label.txt"
    #data_file = "/disk2/data/ava_test/marks.log"
    Std2Tf(dataset_dir, label_file, data_file)


if __name__ == '__main__':
    test(sys.argv[1])
