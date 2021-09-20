import cv2
import argparse
import numpy as np
import tensorflow as tf
from cityscapes_utiles import LABEL_LEGEND


def dataset_mapper(file_path):
    input_image = tf.io.read_file(file_path)
    input_image = tf.image.decode_png(input_image, channels=3)
    input_image = tf.image.resize(input_image, [1024, 2048])
    input_image = tf.cast(input_image, tf.float32) / 255
    return file_path, input_image


def prediction2color(prediction: np.array):
    index_map = np.argmax(prediction, axis=2)  # max index at each pix
    colored = np.zeros(shape=(*index_map.shape, 3), dtype=np.uint8)
    for class_id in LABEL_LEGEND:
        selection = index_map == class_id
        colored[selection] = LABEL_LEGEND[class_id]['color']
    return colored


def main():
    parser = argparse.ArgumentParser(description='Inference and visualization of segmentation')
    parser.add_argument('--model', help='path to TensorFlow/Keras model')
    parser.add_argument('--images', nargs='+', help='path to images with wildcards')
    args = parser.parse_args()

    dataset = tf.data.Dataset.from_tensor_slices(args.images)
    dataset = dataset.map(dataset_mapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=1)

    model = tf.keras.models.load_model(args.model)

    for path, image in dataset:
        mask = model.predict(image)
        mask = prediction2color(mask[0])
        image = (255 * image[0].numpy()).astype(np.uint8)

        display = image//2 + mask//2
        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        path = path.numpy()[0].decode('ASCII')

        cv2.imshow(path, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
