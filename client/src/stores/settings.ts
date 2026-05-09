import { defineStore } from "pinia";
import { ref } from "vue";

export const useSettingsStore = defineStore("settings", () => {
  const detectionThreshold = ref(0.5);
  const overlapThreshold = ref(0.4);
  const thresholdIou = ref(0.2);
  const vehicleOnly = ref(true);
  const allowWebGPU = ref(true);
  const showDetections = ref(true);
  const fps = ref(1);
  const processing = ref(true);
  const evaluateTime = ref(5000);
  const autoEvaluateTime = ref(10000);
  const autoDetect = ref(false);

  return {
    detectionThreshold,
    overlapThreshold,
    thresholdIou,
    vehicleOnly,
    allowWebGPU,
    showDetections,
    fps,
    processing,
    evaluateTime,
    autoEvaluateTime,
    autoDetect,
  };
});
