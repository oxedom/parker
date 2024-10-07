import { defineStore } from 'pinia'
import * as tf from '@tensorflow/tfjs'
import { ref } from 'vue'
export const useEngineStore = defineStore('engine', () => {
  const model = ref<any>(null)

  const initYoloModel = async () => {
    let yolov7 = await tf.loadGraphModel(`${window.location.origin}/yolov7_web_model/model.json`, {
      onProgress: (fractions) => {
        //Loading
        console.log(`Yolo7 model loaded: ${fractions * 100}%`)
      }
    })

    model.value = yolov7

    console.log('Yolo7 model loaded', yolov7)
  }

  return { initYoloModel, model }
})
