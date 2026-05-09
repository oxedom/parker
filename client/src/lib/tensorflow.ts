import * as tf from "@tensorflow/tfjs";
import { xywh2xyxy } from "@/lib/geometry";

export function processInputImage(
  source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
  modelDim: [number, number],
): tf.Tensor {
  return tf.tidy(() =>
    tf.image
      .resizeBilinear(tf.browser.fromPixels(source), modelDim)
      .div(255.0)
      .transpose([2, 0, 1])
      .expandDims(0),
  );
}

export function nonMaxSuppression(
  res: number[][],
  confThresh = 0.5,
  iouThresh = 0.2,
): number[][] {
  const selected: number[][] = [];

  for (let i = 0; i < res.length; i++) {
    if (res[i][4] < confThresh) continue;

    const box = res[i].slice(0, 4);
    const clsScores = res[i].slice(5);
    const klass = clsScores.reduce((imax, x, idx, arr) => (x > arr[imax] ? idx : imax), 0);
    const score = res[i][klass + 5];

    const obj = xywh2xyxy(box);
    let keep = true;

    for (const other of selected) {
      const sb = xywh2xyxy(other);
      const ix1 = Math.max(obj[0], sb[0]);
      const iy1 = Math.max(obj[1], sb[1]);
      const ix2 = Math.min(obj[2], sb[2]);
      const iy2 = Math.min(obj[3], sb[3]);
      const iw = Math.max(0, ix2 - ix1);
      const ih = Math.max(0, iy2 - iy1);
      const inter = iw * ih;
      const aArea = (obj[2] - obj[0]) * (obj[3] - obj[1]);
      const bArea = (sb[2] - sb[0]) * (sb[3] - sb[1]);
      const iou = inter / (aArea + bArea - inter);

      if (iou >= iouThresh) {
        keep = false;
        break;
      }
    }

    if (keep) selected.push([...box, score, klass]);
  }

  return selected;
}
