import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def make_gradcam_heatmap_1d(img_array, model, last_conv_layer_name="last_conv"):
    """Generate a 1-D Grad-CAM heatmap for the model's sigmoid AFib output."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, max_value)
    return heatmap.numpy()
