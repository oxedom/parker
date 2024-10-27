<template>
  <RouterView />
</template>

<script setup lang="ts">
import { useEngineStore } from './stores/engine'
import { onMounted } from 'vue'

const engine = useEngineStore()

onMounted(() => {
  engine.initYoloModel()
})

import './index.css'
</script>

<style scoped></style>
