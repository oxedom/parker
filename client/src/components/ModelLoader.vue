<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useDimensionsStore } from "@/stores/dimensions";

const props = defineProps<{ progress: number }>();

const dims = useDimensionsStore();
const canvasRef = ref<HTMLCanvasElement | null>(null);

function render() {
  const c = canvasRef.value;
  if (!c) return;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  const w = dims.imageWidth;
  const h = dims.imageHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.font = "bold 32px Arial";
  ctx.fillText("Loading Model...", w * 0.5, h * 0.3);
  ctx.fillText(`${Math.round(props.progress * 100)}%`, w * 0.5, h * 0.5);
}

onMounted(render);
watch(() => [props.progress, dims.imageWidth, dims.imageHeight], render);
</script>

<template>
  <canvas ref="canvasRef" :width="dims.imageWidth" :height="dims.imageHeight" />
</template>
