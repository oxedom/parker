<script setup lang="ts">
import { ref, watch } from "vue";
import { useSettingsStore } from "@/stores/settings";
import { useDimensionsStore } from "@/stores/dimensions";
import AppAccordion from "@/components/AppAccordion.vue";
import AppButton from "@/components/AppButton.vue";
import AppSlider from "@/components/AppSlider.vue";
import ToggleSwitch from "@/components/ToggleSwitch.vue";

const settings = useSettingsStore();
const dims = useDimensionsStore();

const localDetectionThreshold = ref(settings.detectionThreshold * 100);
const localIouThreshold = ref(settings.thresholdIou * 100);
const localFps = ref(settings.fps);
const localVehicleOnly = ref(settings.vehicleOnly);
const dirty = ref(false);

watch([localDetectionThreshold, localIouThreshold, localFps, localVehicleOnly], () => {
  dirty.value = true;
});

function applySettings() {
  if (!dirty.value) return;
  settings.detectionThreshold = localDetectionThreshold.value / 100;
  settings.thresholdIou = localIouThreshold.value / 100;
  settings.fps = localFps.value;
  settings.vehicleOnly = localVehicleOnly.value;
  // Trigger detection loop reset by toggling processing flag.
  settings.processing = false;
  dirty.value = false;
  setTimeout(() => {
    settings.processing = true;
  }, 10);
}
</script>

<template>
  <div
    class="md:w-[200px] flex justify-between rounded-xl flex-col bg-black/60 backdrop-blur-sm"
    :style="{ minHeight: dims.imageHeight + 'px' }"
  >
    <AppAccordion title="Settings">
      <div class="flex flex-col gap-3">
        <ToggleSwitch v-model="settings.processing" text="TFJS" />
        <ToggleSwitch v-model="settings.showDetections" text="Show Boxes" />
        <ToggleSwitch v-model="settings.allowWebGPU" text="WebGPU" />
        <ToggleSwitch v-model="localVehicleOnly" text="Vehicle Only" />
      </div>

      <AppSlider
        v-model="localDetectionThreshold"
        label="Detection Threshold"
      />
      <AppSlider v-model="localIouThreshold" label="IOU Threshold" />
      <AppSlider
        v-model="localFps"
        label="Execute Rate"
        :max="2.1"
        :min="0.01"
        :step="0.1"
        unit="s"
      />

      <AppButton class="mb-2" intent="primary" full-width @click="applySettings">
        Apply settings
      </AppButton>
    </AppAccordion>
  </div>
</template>
