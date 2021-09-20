import argparse
import numpy as np
import tensorflow as tf

from mobilenet_unet import unet_model
from cityscapes_utiles import match_filenames, list_classes
from cityscapes_utiles import LABEL_LEGEND, COLOR_SET_RGB

INPUT_SHAPE = [1024, 2048, 3]
CLASSES_LIST = []


def dataset_parameterize(tf_data: tf.data.Dataset, count: int, mapper,
                         batch_size: int, test_scale: float):
    prep_dataset = tf_data.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    prep_dataset = prep_dataset.shuffle(count//100, reshuffle_each_iteration=False)
    prep_dataset = prep_dataset.batch(batch_size)
    train_size = int(count * (1 - test_scale))
    test_size = int(count * test_scale)
    train_tf_dateset = prep_dataset.take(train_size)
    test_tf_dataset = prep_dataset.skip(train_size).take(test_size)
    return train_tf_dateset, test_tf_dataset


def dataset_mapper(rgb_image_path, gt_image_path):
    input_image = tf.io.read_file(rgb_image_path)
    input_image = tf.image.decode_png(input_image, channels=3)
    input_image = tf.image.resize(input_image, INPUT_SHAPE[:2])
    input_image = tf.cast(input_image, tf.float32) / 255
    output_image = tf.io.read_file(gt_image_path)
    output_image = tf.image.decode_png(output_image, channels=3)
    output_image = tf.image.resize(output_image, size=INPUT_SHAPE[:2],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_image = output_image[:, :, 0]
    zeros = tf.zeros(shape=INPUT_SHAPE[:2], dtype=output_image.dtype)
    if CLASSES_LIST:
        for index in LABEL_LEGEND:
            if index in CLASSES_LIST:
                continue
            condition = tf.equal(output_image, index)
            output_image = tf.where(condition, zeros, output_image)
    output_image = output_image[..., tf.newaxis]
    output_image = tf.cast(output_image, tf.float32)
    return input_image, output_image


def frozen_model_export(tf_model: tf.keras.Model, path: str):
    concrete_function = tf.TensorSpec(tf_model.inputs[0].shape, tf_model.inputs[0].dtype)
    full_model = tf.function(lambda x: tf_model(x))
    full_model = full_model.get_concrete_function(x=concrete_function)
    var_converter = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2
    frozen_func = var_converter(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir='', name=path, as_text=False)


def main():
    global CLASSES_LIST
    parser = argparse.ArgumentParser(description='Inference and visualization of segmentation')
    parser.add_argument('--logdir', help='path to TensorFlow/Keras model log folder')
    parser.add_argument('--input', nargs='+', help='path to input images with wildcards')
    parser.add_argument('--output', nargs='+', help='path to output images with wildcards')
    parser.add_argument('--classes', nargs='+', type=int, help='list of class indexes to train on')
    parser.add_argument('--base', type=bool, default=True, help='base network trainable')
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--batch', type=int, default=1, help='')
    args = parser.parse_args()

    CLASSES_LIST = args.classes if args.classes else list()
    args.input, args.output = match_filenames(args.input, args.output)
    tf_dataset = tf.data.Dataset.from_tensor_slices((args.input, args.output))
    train, test = dataset_parameterize(tf_dataset, len(args.input), dataset_mapper, batch_size=args.batch, test_scale=0.1)

    # for image, mask in train:
    #     print(mask.dtype, mask.shape)

    # cls_num = len(CLASSES_LIST) if CLASSES_LIST else len(LABEL_LEGEND)
    segmentation_model = unet_model(input_shape=INPUT_SHAPE, base_trainable=args.base)
    # for layer in segmentation_model.layers: layer.trainable = True

    segmentation_model.summary()
    segmentation_model.compile(optimizer='adam', metrics=['accuracy'],
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # tf.keras.utils.plot_model(segmentation_model, show_shapes=True)
    history = segmentation_model.fit(train, epochs=args.epochs, validation_data=test)
    segmentation_model.save(args.logdir)
    # frozen_model_export(segmentation_model, 'frozen_graph.pb')


if __name__ == '__main__':
    main()
