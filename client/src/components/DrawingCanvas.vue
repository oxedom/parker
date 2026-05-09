<script setup lang="ts">
import { ref } from "vue";
import { storeToRefs } from "pinia";
import { useDimensionsStore } from "@/stores/dimensions";
import { useRoisStore } from "@/stores/rois";
import { useDrawingCanvas } from "@/composables/useDrawingCanvas";

const dims = useDimensionsStore();
const rois = useRoisStore();

const { imageWidth, imageHeight } = storeToRefs(dims);
const { items } = storeToRefs(rois);

const drawCanvas = ref<HTMLCanvasElement | null>(null);
const overlayCanvas = ref<HTMLCanvasElement | null>(null);

const { isDown, onMouseDown, onMouseMove, onMouseUp, onMouseOut } =
  useDrawingCanvas({
    drawCanvas,
    overlayCanvas,
    rois: items,
    imageWidth,
    imageHeight,
    onRoiCreated: (cords) => rois.add(cords),
  });
</script>

<template>
  <div :class="isDown ? 'cursor-grabbing' : 'cursor-grab'">
    <canvas
      ref="drawCanvas"
      class="fixed"
      :style="{ zIndex: 2 }"
      :width="dims.imageWidth"
      :height="dims.imageHeight"
    />
    <canvas
      ref="overlayCanvas"
      class="fixed"
      :style="{ zIndex: 3 }"
      :width="dims.imageWidth"
      :height="dims.imageHeight"
      id="draw_canvas"
      @mousedown="onMouseDown"
      @mousemove="onMouseMove"
      @mouseup="onMouseUp"
      @mouseout="onMouseOut"
    />
  </div>
</template>
