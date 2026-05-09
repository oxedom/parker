<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useDimensionsStore } from "@/stores/dimensions";

const props = defineProps<{
  allowWebcam: boolean;
  webRtcMode: boolean;
}>();

const dims = useDimensionsStore();
const canvasRef = ref<HTMLCanvasElement | null>(null);

const message = computed<string>(() => {
  if (props.allowWebcam) return "Attempting to detect webcam...";
  if (props.webRtcMode)
    return "Invite with link\nto make a remote\nvideo connection";
  return "Choose a video source from\nthe navigation bar\n\nDemo / Webcam / Remote";
});

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
  ctx.font = "bold 32px Arial";
  ctx.textAlign = "center";
  const lines = message.value.split("\n");
  const lh = 40;
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], w / 2, h / 2.5 + i * lh);
  }
}

onMounted(render);
watch(
  () => [props.allowWebcam, props.webRtcMode, dims.imageWidth, dims.imageHeight],
  render,
);
</script>

<template>
  <canvas
    ref="canvasRef"
    class="z-10 relative"
    :width="dims.imageWidth"
    :height="dims.imageHeight"
  />
</template>
