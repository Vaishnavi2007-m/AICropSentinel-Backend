import numpy as np
import tensorflow as tf
import cv2

def get_gradcam(img_array, model, last_conv_layer_name):

    # Create model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.outputs
        ]
    )

    # Gradient calculation
    with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(img_array)

      preds = predictions[0].numpy()      # convert safely
      predicted_class = np.argmax(preds)  # always works

      loss = predictions[0][predicted_class]
    # Get gradients
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    
    # Heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap.numpy()