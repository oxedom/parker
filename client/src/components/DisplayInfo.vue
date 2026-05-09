<script setup lang="ts">
import { useRoisStore } from "@/stores/rois";

const rois = useRoisStore();
</script>

<template>
  <div
    class="flex gap-5 justify-center items-center place-content-center self-center"
  >
    <div class="flex items-center flex-col gap-2">
      <p class="border-b-4 border-green-700 rounded">Available</p>
      <span>{{ rois.counts.available }}</span>
    </div>
    <div class="flex items-center flex-col gap-2">
      <p class="border-b-4 border-red-700 rounded">Occupied</p>
      <span>{{ rois.counts.occupied }}</span>
    </div>
  </div>
</template>
