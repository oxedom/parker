import * as tf from "@tensorflow/tfjs";
import { ref, shallowRef } from "vue";
import { processInputImage, nonMaxSuppression } from "@/lib/tensorflow";

const MODEL_DIM: [number, number] = [640, 640];

export interface DetectorOptions {
  modelUrl?: string;
  allowWebGPU?: boolean;
  onProgress?: (fraction: number) => void;
}

export function useDetector() {
  const ready = ref(false);
  const progress = ref(0);
  const error = ref<unknown>(null);
  const model = shallowRef<tf.GraphModel | null>(null);

  async function setBackend(allowWebGPU: boolean) {
    if (allowWebGPU) {
      await import("@tensorflow/tfjs-backend-webgl");
      try {
        await tf.setBackend("webgl");
      } catch (e) {
        console.warn("WebGL backend unavailable, falling back to CPU", e);
        await import("@tensorflow/tfjs-backend-cpu");
        await tf.setBackend("cpu");
      }
    } else {
      await import("@tensorflow/tfjs-backend-cpu");
      await tf.setBackend("cpu");
    }
    await tf.ready();
  }

  async function load(opts: DetectorOptions = {}) {
    if (model.value) return;
    try {
      await setBackend(opts.allowWebGPU ?? true);
      const url =
        opts.modelUrl ?? `${import.meta.env.BASE_URL}yolov7_web_model/model.json`;

      const m = await tf.loadGraphModel(url, {
        onProgress: (f) => {
          progress.value = f;
          opts.onProgress?.(f);
        },
      });

      // Warm-up
      const dummy = tf.ones(m.inputs[0].shape as number[]);
      const warm = await m.executeAsync(dummy);
      tf.dispose(warm);
      tf.dispose(dummy);

      model.value = m;
      ready.value = true;
    } catch (e) {
      error.value = e;
      console.error("Failed to load model", e);
    }
  }

  async function detect(
    source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
    detectionThreshold: number,
    iouThreshold: number,
  ): Promise<number[][]> {
    if (!model.value) return [];

    tf.engine().startScope();
    try {
      const input = processInputImage(source, MODEL_DIM);
      const res = model.value.execute(input) as tf.Tensor;
      const arr = (res.arraySync() as number[][][])[0];
      const detections = nonMaxSuppression(arr, detectionThreshold, iouThreshold);
      tf.dispose(input);
      tf.dispose(res);
      return detections;
    } finally {
      tf.engine().endScope();
    }
  }

  function dispose() {
    model.value?.dispose();
    model.value = null;
    ready.value = false;
  }

  return { ready, progress, error, load, detect, dispose, setBackend };
}
