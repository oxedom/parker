<template>

    <div :style="cSizeStyles" class="relative">
        <video :style="cSizeStyles" class="absolute inset-0 object-cover" ref="video" muted loop src="/demo.mp4"
            autoplay />
        <canvas :style="cSizeStyles" class="absolute inset-0" ref="canvas"></canvas>
    </div>

</template>

<script setup lang="ts">


// const props = defineProps({
//     foo: { type: String, required: true },
//     bar: Number
// })

import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import { useEngineStore } from '../stores/engine'
import { handleRendering } from '../utils/canvas'

const id = ref(0)
const video = ref(null)
const containerWidth = ref(600)
const containerHeight = ref(300)

const cSizeStyles = computed(() => {
    return {
        width: `${containerWidth.value}px`,
        height: `${containerHeight.value}px`
    }
})


const engine = useEngineStore()

onMounted(() => {
    window.setInterval(() => {
        try {
            const result = engine.processFrame(video.value)
            handleRendering(result)
        } catch (e) { }
    }, 2000)
})

onBeforeUnmount(() => {
    clearInterval(id.value)
})




</script>

<style></style>