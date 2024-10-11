import { defineStore } from 'pinia'
import * as tf from '@tensorflow/tfjs'
import { ref } from 'vue'
import { processInputImage, non_max_suppression } from '@/utils/tfjs.js'
import { detectionsToROIArr } from '@/utils/overlap.js'
import '@tensorflow/tfjs-backend-webgl' //TODO Check that this is required

export const useEngineStore = defineStore('engine', () => {
  const model = ref<any>(null)

  const initYoloModel = async () => {
    let yolov7 = await tf.loadGraphModel('/yolov7_web_model/model.json', {
      onProgress: (fractions) => {
        //Loading
        console.log(`Yolo7 model loaded: ${fractions * 100}%`)
      }
    })

    model.value = Object.freeze(yolov7)
    const tensor = yolov7?.inputs?.[0]?.shape
    if (tensor) {
      const dummyInput = tf.ones(tensor)
      const warmupResult = await yolov7.executeAsync(dummyInput)
      tf.dispose(warmupResult)
      tf.dispose(dummyInput)
    }
  }

  const processFrame = async (videoElement: HTMLVideoElement) => {
    if (!model.value || !videoElement) return

    tf.engine().startScope()
    let input = processInputImage(videoElement, [640, 640])

    let res = model.value.execute(input)
    const detections = non_max_suppression(res.arraySync()[0], 0.5, 0.2, 50)

    const vehicleOnly = true

    const predictionsRois = detectionsToROIArr(
      detections,
      videoElement.videoWidth,
      videoElement.videoHeight,
      vehicleOnly
    )

    console.log(predictionsRois)

    try {
      await Promise.all([tf.dispose(input), tf.dispose(res)])
    } catch (err) {
      console.warn('🚀 ~ processFrame - Memory LEAK ~ err:', err)
    }

    tf.engine().endScope()
  }

  return { initYoloModel, model, processFrame }
})
