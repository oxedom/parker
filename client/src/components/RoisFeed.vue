<script setup lang="ts">
import { useDimensionsStore } from "@/stores/dimensions";
import { useRoisStore } from "@/stores/rois";
import AppAccordion from "@/components/AppAccordion.vue";
import AppButton from "@/components/AppButton.vue";

const emit = defineEmits<{ openModal: [] }>();

const rois = useRoisStore();
const dims = useDimensionsStore();
const baseUrl = import.meta.env.BASE_URL;

function handleDeleteAll() {
  if (rois.items.length === 0) return;
  if (window.confirm("Are you sure you want to delete all selected regions?")) {
    rois.clear();
  }
}
</script>

<template>
  <div
    class="w-[200px] bg-black/60 backdrop-blur-sm rounded-xl"
    :style="{ minHeight: dims.imageHeight + 'px' }"
  >
    <AppAccordion title="Controls">
      <div class="flex flex-col gap-2 my-2">
        <AppButton intent="destructive" @click="handleDeleteAll">
          Delete regions
        </AppButton>
        <AppButton @click="rois.startAutoDetect()">
          Auto Detect (Beta)
        </AppButton>
        <AppButton @click="emit('openModal')">Manage Data</AppButton>
      </div>
    </AppAccordion>

    <h4
      class="text-xl font-semibold text-center text-white border-b-2 border-orange-600 hover:cursor-default"
    >
      Marked regions
    </h4>

    <div class="flex flex-wrap gap-2 m-2">
      <div
        v-for="r in rois.items"
        :key="r.uid"
        class="h-10 w-10 font-semibold hover:bg-yellow-500 transition-colors rounded border border-gray-500 cursor-default duration-100 items-center justify-between"
        :class="{
          'bg-gray-400 animate-pulse duration-1000': r.evaluating,
          'bg-red-500': r.occupied && !r.evaluating,
          'bg-green-500': !r.occupied && !r.evaluating,
        }"
        @mouseover="rois.setHover(r.uid, true)"
        @mouseleave="rois.setHover(r.uid, false)"
        @click="rois.remove(r.uid)"
      >
        <img
          v-if="r.hover"
          :src="`${baseUrl}static/icons/delete_bin_black.png`"
          width="40"
          height="40"
          alt="Delete"
          class="invert opacity-90"
        />
      </div>
    </div>
  </div>
</template>
