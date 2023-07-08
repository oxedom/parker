import * as tf from "@tensorflow/tfjs";
import { xywh2xyxy } from "./canvas_utility";

export function processInputImage(video, model_dim) {
  let input = tf.tidy(() => {
    try {
      return tf.image
        .resizeBilinear(tf.browser.fromPixels(video), model_dim)
        .div(255.0)
        .transpose([2, 0, 1])
        .expandDims(0);
    } catch (error) {
      console.error(error);
    }
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
    //Need to return a 2d tensor and not that
    return tf.zeros([1, 1]);
  }
  try {
    _nmsDetections = await tf.image.nonMaxSuppressionAsync(
      boxes,
      scores,
      100,
      thresholdIou
    );

    _detectionIndices = _nmsDetections.dataSync();
  } catch (error) {
    console.error(error);
  }

  return { detectionIndices: _detectionIndices };
}

//hugoCode
export function non_max_suppression(
  res,
  conf_thresh = 0.5,
  iou_thresh = 0.2,
  max_det = 300
) {
  // Initialize an empty list to store the selected boxes
  const selected_detections = [];

  for (let i = 0; i < res.length; i++) {
    // Check if the box has sufficient score to be selected
    if (res[i][4] < conf_thresh) {
      continue;
    }

    var box = res[i].slice(0, 4);
    const cls_detections = res[i].slice(5);
    var klass = cls_detections.reduce(
      (imax, x, i, arr) => (x > arr[imax] ? i : imax),
      0
    );
    const score = res[i][klass + 5];

    let object = xywh2xyxy(box);
    let addBox = true;

    // Check for overlap with previously selected boxes
    for (let j = 0; j < selected_detections.length; j++) {
      let selectedBox = xywh2xyxy(selected_detections[j]);

      // Calculate the intersection and union of the two boxes
      let intersectionXmin = Math.max(object[0], selectedBox[0]);
      let intersectionYmin = Math.max(object[1], selectedBox[1]);
      let intersectionXmax = Math.min(object[2], selectedBox[2]);
      let intersectionYmax = Math.min(object[3], selectedBox[3]);
      let intersectionWidth = Math.max(0, intersectionXmax - intersectionXmin);
      let intersectionHeight = Math.max(0, intersectionYmax - intersectionYmin);
      let intersectionArea = intersectionWidth * intersectionHeight;
      let boxArea = (object[2] - object[0]) * (object[3] - object[1]);
      let selectedBoxArea =
        (selectedBox[2] - selectedBox[0]) * (selectedBox[3] - selectedBox[1]);
      let unionArea = boxArea + selectedBoxArea - intersectionArea;

      // Calculate the IoU and check if the boxes overlap
      let iou = intersectionArea / unionArea;
      if (iou >= iou_thresh) {
        addBox = false;
        break;
      }
    }

    // Add the box to the selected boxes list if it passed the overlap check
    if (addBox) {
      const row = box.concat(score, klass);
      selected_detections.push(row);
    }
  }

  return selected_detections;
}
