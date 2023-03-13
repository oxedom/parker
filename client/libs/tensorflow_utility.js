import * as tf from "@tensorflow/tfjs";
import { exp } from "@tensorflow/tfjs";

export function processInputImage(video, model_dim) {
  let input = tf.tidy(() => {
    return tf.image
      .resizeBilinear(tf.browser.fromPixels(video), model_dim)
      .div(255.0)
      .transpose([2, 0, 1])
      .expandDims(0);
  });

  return input;
}

export function processDetectionResults(res, detectionThreshold) {
  res = res.arraySync()[0];
  //Filtering only detections > conf_thres
  res = res.filter((dataRow) => dataRow[4] >= detectionThreshold);
  let _boxes = [];
  let _class_detect = [];
  let _scores = [];

  function process_pred(res) {
    let box = res.slice(0, 4);

    const cls_detections = res.slice(5, 85);
    let max_score_index = cls_detections.reduce(
      (imax, x, i, arr) => (x > arr[imax] ? i : imax),
      0
    );

    _boxes.push(box);
    _scores.push(res[max_score_index + 5]);
    _class_detect.push(max_score_index);
  }

  res.forEach(process_pred);

  return { scores: _scores, class_detect: _class_detect, boxes: _boxes };
}

export async function nmsDetectionProcess(boxes, scores, thresholdIou) {
  let _nmsDetections;
  let _detectionIndices;
  if (boxes.length < 0) {
    return [];
  }
  _nmsDetections = await tf.image.nonMaxSuppressionAsync(
    boxes,
    scores,
    100,
    thresholdIou
  );

  _detectionIndices = _nmsDetections.dataSync();

  return { detectionIndices: _detectionIndices };
}

export async function disposeTensors(input, res) {
  console.log("Im disposing");
  return Promise.all([tf.dispose(input), tf.dispose(res)]).catch((err) => {
    console.error("Memory leak in coming");
    console.error(err);
  });
}
