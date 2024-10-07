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
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useEngineStore } from '../stores/engine'
const video = ref<HTMLVideoElement | null>(null)
const engine = useEngineStore()

onMounted(() => {})
</script>

<style></style>
