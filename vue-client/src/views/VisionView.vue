<template>
  <div>
    <video
      ref="video"
      width="600"
      muted
      start="10"
      loop
      src="/public/demo.mp4"
      height="500"
      autoplay
    />

    <button @click="runEngine">Run Yolo</button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useEngineStore } from '../stores/engine'
const video = ref<HTMLVideoElement | null>(null)

const engine = useEngineStore()

console.log(video)

const runEngine = () => engine.runYolo()
</script>

<style></style>
