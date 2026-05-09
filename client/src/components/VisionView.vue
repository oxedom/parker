<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import type { VideoSource } from "@/types";
import { useDimensionsStore } from "@/stores/dimensions";
import { useRoisStore } from "@/stores/rois";
import { useSettingsStore } from "@/stores/settings";
import { useDetector } from "@/composables/useDetector";
import { detectionsToRois } from "@/lib/roi";
import { renderAllOverlaps, drawCenteredText } from "@/lib/canvas";
import ModelLoader from "@/components/ModelLoader.vue";
import LoadingScreen from "@/components/LoadingScreen.vue";

const props = defineProps<{
  source: VideoSource;
  rtcStream: MediaStream | null;
}>();

const emit = defineEmits<{
  videoReady: [{ width: number; height: number; source: VideoSource }];
  rtcLoaded: [];
}>();

const settings = useSettingsStore();
const dims = useDimensionsStore();
const rois = useRoisStore();
const detector = useDetector();

const {
  fps,
  detectionThreshold,
  thresholdIou,
  vehicleOnly,
  showDetections,
  processing,
  allowWebGPU,
  autoDetect,
} = storeToRefs(settings);

const webcamRef = ref<HTMLVideoElement | null>(null);
const demoRef = ref<HTMLVideoElement | null>(null);
const rtcRef = ref<HTMLVideoElement | null>(null);
const overlayRef = ref<HTMLCanvasElement | null>(null);

const webcamStream = ref<MediaStream | null>(null);
const webcamReady = ref(false);
const demoReady = ref(false);
const rtcReady = ref(false);

const showWebcam = computed(() => props.source === "webcam");
const showDemo = computed(() => props.source === "demo");
const showRtc = computed(() => props.source === "rtc");

const showLoadingScreen = computed(
  () =>
    props.source === null ||
    (props.source === "webcam" && !webcamReady.value) ||
    (props.source === "rtc" && !rtcReady.value),
);

let detectionTimer: number | null = null;

function activeVideo(): HTMLVideoElement | null {
  if (showWebcam.value && webcamReady.value) return webcamRef.value;
  if (showDemo.value && demoReady.value) return demoRef.value;
  if (showRtc.value && rtcReady.value) return rtcRef.value;
  return null;
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    webcamStream.value = stream;
    if (webcamRef.value) {
      webcamRef.value.srcObject = stream;
      await webcamRef.value.play().catch(() => {});
    }
  } catch (e) {
    console.error("Webcam error", e);
  }
}

function stopWebcam() {
  webcamStream.value?.getTracks().forEach((t) => t.stop());
  webcamStream.value = null;
  if (webcamRef.value) webcamRef.value.srcObject = null;
  webcamReady.value = false;
}

function onWebcamPlay() {
  const v = webcamRef.value;
  if (!v) return;
  dims.set(v.videoWidth || 640, v.videoHeight || 480);
  webcamReady.value = true;
  emit("videoReady", {
    width: v.videoWidth,
    height: v.videoHeight,
    source: "webcam",
  });
}

function onDemoPlay() {
  const v = demoRef.value;
  if (!v) return;
  dims.set(v.videoWidth || 640, v.videoHeight || 480);
  demoReady.value = true;
  if (rois.items.length === 0) rois.startAutoDetect();
  emit("videoReady", {
    width: v.videoWidth,
    height: v.videoHeight,
    source: "demo",
  });
}

function onRtcPlay() {
  const v = rtcRef.value;
  if (!v) return;
  dims.set(v.videoWidth || 640, v.videoHeight || 480);
  rtcReady.value = true;
  emit("rtcLoaded");
  emit("videoReady", {
    width: v.videoWidth,
    height: v.videoHeight,
    source: "rtc",
  });
}

async function tick() {
  const video = activeVideo();
  const overlay = overlayRef.value;
  if (!video || !detector.ready.value || !overlay) return;
  if (video.readyState < 2 || video.videoWidth === 0) return;

  try {
    const detections = await detector.detect(
      video,
      detectionThreshold.value,
      thresholdIou.value,
    );
    const predictions = detectionsToRois(
      detections,
      dims.imageWidth,
      dims.imageHeight,
      vehicleOnly.value,
    );

    const ctx = overlay.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, dims.imageWidth, dims.imageHeight);
      if (showDetections.value && predictions.length > 0) {
        renderAllOverlaps(predictions, ctx, dims.imageWidth, dims.imageHeight);
      }
      if (autoDetect.value) {
        drawCenteredText(
          ctx,
          dims.imageWidth,
          dims.imageHeight,
          "Auto detecting",
        );
      }
    }

    rois.processFrame({ predictions });
  } catch (e) {
    console.error("Detection tick error", e);
  }
}

function startLoop() {
  stopLoop();
  if (!processing.value) return;
  detectionTimer = window.setInterval(tick, Math.max(10, fps.value * 1000));
}

function stopLoop() {
  if (detectionTimer !== null) {
    clearInterval(detectionTimer);
    detectionTimer = null;
  }
}

watch(
  () => props.source,
  async (newSrc, oldSrc) => {
    if (oldSrc === "webcam" && newSrc !== "webcam") stopWebcam();
    if (newSrc === "webcam" && !webcamStream.value) {
      // wait one tick for the video element to mount
      await Promise.resolve();
      startWebcam();
    }
    if (newSrc === null) {
      demoReady.value = false;
      rtcReady.value = false;
    }
  },
);

watch(
  () => props.rtcStream,
  (stream) => {
    if (stream && rtcRef.value) {
      rtcRef.value.srcObject = stream;
      rtcRef.value.play().catch(() => {});
    }
  },
);

watch(
  [processing, fps, () => detector.ready.value, webcamReady, demoReady, rtcReady],
  () => startLoop(),
);

watch(allowWebGPU, async (val) => {
  await detector.setBackend(val);
});

(async () => {
  await detector.load({ allowWebGPU: allowWebGPU.value });
  startLoop();
})();

onBeforeUnmount(() => {
  stopLoop();
  stopWebcam();
});
</script>

<template>
  <section class="overflow-hidden rounded-md relative">
    <ModelLoader v-if="!detector.ready.value" :progress="detector.progress.value" />

    <template v-else>
      <canvas
        ref="overlayRef"
        :width="dims.imageWidth"
        :height="dims.imageHeight"
        class="fixed"
        id="overlap-overlay"
      />

      <video
        v-show="showWebcam"
        ref="webcamRef"
        muted
        autoplay
        playsinline
        :width="dims.imageWidth"
        :height="dims.imageHeight"
        @loadedmetadata="onWebcamPlay"
      />

      <video
        v-show="showDemo"
        ref="demoRef"
        :width="dims.imageWidth"
        :height="dims.imageHeight"
        muted
        loop
        autoplay
        playsinline
        src="/demo.mp4"
        @play="onDemoPlay"
      />

      <video
        v-show="showRtc && rtcReady"
        ref="rtcRef"
        muted
        autoplay
        playsinline
        :width="dims.imageWidth"
        :height="dims.imageHeight"
        @play="onRtcPlay"
      />

      <LoadingScreen
        v-if="showLoadingScreen"
        :allow-webcam="showWebcam"
        :web-rtc-mode="showRtc"
      />
    </template>
  </section>
</template>
