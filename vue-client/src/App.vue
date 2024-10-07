<template>
  <RouterView />

  <button @click="runYolo">Run Yolo</button>
</template>

<script setup lang="ts">
import './index.css'
import * as tf from '@tensorflow/tfjs'

const runYolo = async () => {
  let yolov7 = await tf.loadGraphModel(`${window.location.origin}/yolov7_web_model/model.json`, {
    onProgress: (fractions) => {
      //Loading
      console.log(`Yolo7 model loaded: ${fractions}%`)
    }
  })

  console.log('Yolo7 model loaded', yolov7)
}
</script>

<style scoped></style>
