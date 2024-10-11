<template>
  <div>
    <video ref="video" muted start="10" loop src="/demo.mp4" autoplay />
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useEngineStore } from '../stores/engine'

const id = ref(0)
const video = ref(null)
const engine = useEngineStore()

onMounted(() => {
  window.setInterval(() => {
    try {
      engine.processFrame(video.value)
    } catch (e) {}
  }, 2000)
})

onBeforeUnmount(() => {
  clearInterval(id.value)
})
</script>

<style></style>
