<script setup lang="ts">
import { ref, watch } from "vue";
import QRCode from "qrcode";
import AppButton from "@/components/AppButton.vue";
import DisplayInfo from "@/components/DisplayInfo.vue";
import type { VideoSource } from "@/types";

const props = defineProps<{
  source: VideoSource;
  peerId: string;
}>();

const emit = defineEmits<{
  setSource: [VideoSource];
}>();

const qrCodeUrl = ref<string>("");

const btnClass = (active: boolean) =>
  `border rounded-xl m-2 text-xl p-2 shadow-sm shadow-black text-center hover:scale-105 duration-200 ${
    active ? "" : "animate-pulse"
  } hover:shadow-none`;

async function regenerateQR(id: string) {
  if (!id) {
    qrCodeUrl.value = "";
    return;
  }
  try {
    const url = new URL(window.location.href);
    url.pathname = "/reroute";
    url.searchParams.set("remoteID", id);
    qrCodeUrl.value = await QRCode.toDataURL(url.href);
  } catch (e) {
    console.error("QR generation failed", e);
  }
}

watch(() => props.peerId, regenerateQR, { immediate: true });

function handleCopy() {
  if (!props.peerId) return;
  const url = new URL(window.location.href);
  url.pathname = "/reroute";
  url.searchParams.set("remoteID", props.peerId);
  navigator.clipboard.writeText(url.href);
}
</script>

<template>
  <nav
    class="flex justify-between items-center animate-fade bg-black/60 backdrop-blur-sm rounded-md"
  >
    <div
      v-if="source === null"
      class="flex flex-row items-center justify-between w-full ml-3 mr-6"
    >
      <section>
        <button
          :class="`bg-purple-600 ${btnClass(false)}`"
          @click="emit('setSource', 'demo')"
        >
          Video Demo
        </button>
        <button
          :class="`bg-purple-600 ${btnClass(false)}`"
          @click="emit('setSource', 'webcam')"
        >
          Webcam
        </button>
        <button
          :class="`bg-purple-600 ${btnClass(false)}`"
          @click="emit('setSource', 'rtc')"
        >
          Remote
        </button>
      </section>
    </div>
    <div
      v-else
      class="flex items-center justify-between w-full grid-cols-3 gap-10 px-3"
    >
      <AppButton intent="destructive" @click="emit('setSource', null)">
        Back
      </AppButton>
      <DisplayInfo />
      <section class="relative group">
        <AppButton @click="handleCopy">Copy Link</AppButton>
        <img
          v-if="qrCodeUrl"
          :src="qrCodeUrl"
          alt="QR code"
          width="80"
          height="75"
          class="scale-[2] hidden group-hover:block duration-200 absolute bottom-[5rem] left-1/2 -translate-x-1/2"
        />
      </section>
    </div>
  </nav>
</template>
