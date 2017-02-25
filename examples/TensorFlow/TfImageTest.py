import tensorflow as tf


def ImageTest(path):
    with tf.Session() as sess:
        img0 = tf.read_file(path)
        image0 = tf.image.decode_image(img0)
        image1 = tf.image.decode_jpeg(img0)

        image0.set_shape([255, 255, 3])
        resized_image = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.AREA)

        img0, image0, image1, resized_image= sess.run([img0, image0, image1, resized_image])
        #assertAllEqual(image0, image1)
    return img0, image0, image1, resized_image


def ImageTest1(path):
    with tf.Session("") as sess:
        try:

            _decode_jpeg_data = tf.placeholder(dtype=tf.string)
            _decode_jpeg = tf.image.decode_image(_decode_jpeg_data, channels=3)
            img0 = open(path).read()
            img1 = sess.run(_decode_jpeg, feed_dict={_decode_jpeg_data: img0})
            return True, img1
        except Exception as e:
            #print("Test Error: " + path + str(e))
            return False, None

def imglist_test():
    file_list = ["/disk2/data/flowers/data/flower_photos/daisy/17101762155_2577a28395.jpg", \
        "/disk2/data/imagetest/test.jpg", \
        "/disk2/data/imagetest/test.png", \
        "/disk2/data/imagetest/test.tiff", \
        "/disk2/data/imagetest/test.gif", \
        "/disk2/data/imagetest/test.txt", \
        "/disk2/data/imagetest/test.jp" ]

    for f in file_list:
        try:
            res, image = ImageTest1(f)
            assert len(image.shape) == 3
            assert image.shape[2] == 3
            print image.shape
        except Exception as e:
            print("Error Occur: " + f)



reader = tf.WholeFileReader()

key, value = reader.read(tf.train.string_input_producer(['/disk2/data/flowers/data/flower_photos/daisy/17101762155_2577a28395.jpg']))
img0 = tf.image.decode_jpeg(value)
img = tf.expand_dims(img0, 0)
image_summary = tf.summary.image("original image", img)
histogram_summary = tf.summary.histogram('image hist', img)
e = tf.reduce_mean(img)
scalar_summary = tf.summary.scalar("image mean", e)

# resize
resized_image = tf.image.resize_images(img, [256, 256], method=tf.image.ResizeMethod.AREA)
img_resize_summary = tf.summary.image('image resized', resized_image)
# crop
cropped_image = tf.image.crop_to_bounding_box(img0, 20, 20, 256, 256)
cropped_image_summary = tf.summary.image('image cropped', tf.expand_dims(cropped_image, 0))
# flip
flipped_image = tf.image.flip_left_right(img0)
flipped_image_summary = tf.summary.image('image flipped', tf.expand_dims(flipped_image, 0))
# grey
grayed_image = tf.image.rgb_to_grayscale(img0)
grayed_image_summary = tf.summary.image('image grayed', tf.expand_dims(grayed_image, 0))
# rotate
rotated_image = tf.image.rot90(img0, k=1)
rotated_image_summary = tf.summary.image('image rotated', tf.expand_dims(rotated_image, 0))

merged = tf.summary.merge_all()

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  print(sess.run(init_op))
  cord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=cord)
  image = img.eval()
  print(image.shape)
  cord.request_stop()
  cord.join(threads)
  summary_writer = tf.summary.FileWriter('/tmp/tensorboard', sess.graph)
  summary_all = sess.run(merged)
  summary_writer.add_summary(summary_all, 0)
  summary_writer.close()
