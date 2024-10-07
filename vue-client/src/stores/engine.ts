import { defineStore } from 'pinia'
import * as tf from '@tensorflow/tfjs'
export const useEngineStore = defineStore('counter', () => {
  const runYolo = async () => {
    let yolov7 = await tf.loadGraphModel(`${window.location.origin}/yolov7_web_model/model.json`, {
      onProgress: (fractions) => {
        //Loading
        console.log(`Yolo7 model loaded: ${fractions * 100}%`)
      }
    })

    console.log('Yolo7 model loaded', yolov7)
  }

  return { runYolo }
})
