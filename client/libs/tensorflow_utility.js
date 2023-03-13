import * as tf from "@tensorflow/tfjs";

export function processInputImage(video, model_dim) 
{
  let input = tf.tidy(() => {
    return tf.image
      .resizeBilinear(tf.browser.fromPixels(video), model_dim)
      .div(255.0)
      .transpose([2, 0, 1])
      .expandDims(0);
  });

  return input
}