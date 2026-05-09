<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import DashboardLayout from "@/components/DashboardLayout.vue";
import VisionHeader from "@/components/VisionHeader.vue";
import VisionToolbar from "@/components/VisionToolbar.vue";
import VisionView from "@/components/VisionView.vue";
import RoisFeed from "@/components/RoisFeed.vue";
import DrawingCanvas from "@/components/DrawingCanvas.vue";
import AppModal from "@/components/AppModal.vue";
import DataManager from "@/components/DataManager.vue";
import { useDimensionsStore } from "@/stores/dimensions";
import { useRoisStore } from "@/stores/rois";
import { usePeer } from "@/composables/usePeer";
import { createEmptyStream } from "@/lib/webrtc";
import type { VideoSource } from "@/types";

const dims = useDimensionsStore();
const rois = useRoisStore();
const peer = usePeer();
const { items } = storeToRefs(rois);

const source = ref<VideoSource>(null);
const rtcStream = ref<MediaStream | null>(null);
const rtcLoaded = ref(false);
const isModalOpen = ref(false);

const showOverlayCanvases = computed(
  () =>
    (source.value === "demo" && true) ||
    (source.value === "webcam" && true) ||
    (source.value === "rtc" && rtcLoaded.value),
);

watch(items, (next) => {
  if (next.length > 0) peer.send(next);
}, { deep: true });

function setSource(next: VideoSource) {
  source.value = next;
  rtcLoaded.value = false;
  if (next === null) {
    dims.set(640, 480);
  }
}

onMounted(async () => {
  await peer.init();
  peer.onIncomingStream((stream) => {
    rtcStream.value = stream;
  });
  // accept calls with an empty media stream so the connection is established
  watch(
    () => peer.peer.value,
    (p) => {
      if (!p) return;
      p.on("call", (call) => {
        call.answer(createEmptyStream());
      });
    },
    { immediate: true },
  );
});
</script>

<template>
  <DashboardLayout>
    <AppModal :is-open="isModalOpen" @close="isModalOpen = false">
      <DataManager />
    </AppModal>

    <div class="flex flex-col">
      <div class="flex flex-col items-center justify-center gap-4 rounded-lg">
        <div
          class="relative grid w-full h-20 gap-2 text-2xl font-bold text-white rounded-md"
        >
          <VisionHeader
            :source="source"
            :peer-id="peer.peerId.value"
            @set-source="setSource"
          />
        </div>

        <div
          class="flex-col hidden gap-4 md:flex md:flex-row md:justify-between"
        >
          <VisionToolbar />

          <div>
            <DrawingCanvas v-if="showOverlayCanvases" />
            <VisionView
              :source="source"
              :rtc-stream="rtcStream"
              @rtc-loaded="rtcLoaded = true"
            />
          </div>

          <RoisFeed @open-modal="isModalOpen = true" />
        </div>
      </div>
    </div>
  </DashboardLayout>
</template>
