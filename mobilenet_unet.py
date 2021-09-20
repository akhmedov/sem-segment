import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def encoder(input_shape, base_trainable: bool):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = base_trainable
    return down_stack


def decoder():
    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3),
    ]
    return up_stack


def unet_model(input_shape, base_trainable: bool):
    inputs = tf.keras.layers.Input(shape=input_shape)
    skips = encoder(input_shape=input_shape, base_trainable=base_trainable)(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # Up-sampling and establishing the skip connections
    for up, skip in zip(decoder(), skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(padding='same', kernel_size=3, strides=2, filters=35)
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
