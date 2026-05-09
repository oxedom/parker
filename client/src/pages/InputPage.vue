<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import type { MediaConnection } from "peerjs";
import { usePeer } from "@/composables/usePeer";

const route = useRoute();
const peer = usePeer();

const inputRef = ref<HTMLVideoElement | null>(null);
const stream = ref<MediaStream | null>(null);
const callRef = ref<MediaConnection | null>(null);
const connected = ref(false);

onMounted(async () => {
  try {
    const { default: NoSleep } = await import("nosleep.js");
    await new NoSleep().enable();
  } catch (e) {
    // Wake Lock requires a user gesture in many browsers; failure is non-fatal.
    console.warn("NoSleep unavailable", e);
  }
  await peer.init();
});

onBeforeUnmount(hangup);

async function shareVideo(): Promise<MediaStream | null> {
  try {
    const s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "environment" } },
    });
    stream.value = s;
    if (inputRef.value) inputRef.value.srcObject = s;
    return s;
  } catch {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      stream.value = s;
      if (inputRef.value) inputRef.value.srcObject = s;
      return s;
    } catch (e) {
      console.error(e);
      return null;
    }
  }
}

async function call() {
  const remoteId = (route.query.remoteID as string) ?? "";
  if (!remoteId) return;

  const s = await shareVideo();
  if (!s) return;

  const c = peer.call(remoteId, s);
  callRef.value = c;
  c?.on("stream", () => {
    connected.value = true;
  });
}

function hangup() {
  callRef.value?.close();
  callRef.value = null;
  connected.value = false;
  stream.value?.getTracks().forEach((t) => t.stop());
  stream.value = null;
  if (inputRef.value) inputRef.value.srcObject = null;
}
</script>

<template>
  <div>
    <div
      class="h-screen gap-2 pt-10 flex flex-col min-h-screen bg-fixed bg-no-repeat bg-cover bg-filler w-full grow items-center"
    >
      <p class="text-5xl py-2 text-white">
        Connection: {{ connected ? "Established" : "Pending" }}
      </p>
      <div class="flex flex-col md:flex-row gap-4">
        <button
          class="bg-green-400 py-2 rounded-lg shadow-sm active:bg-green-600 hover:bg-green-500 text-white font-bold text-4xl p-5 w-[250px]"
          @click="call"
        >
          Call
        </button>
        <button
          class="bg-red-400 py-2 rounded-lg shadow-sm active:bg-red-600 hover:bg-red-500 text-white font-bold text-4xl p-5 w-[250px]"
          @click="hangup"
        >
          Hang up
        </button>
        <video ref="inputRef" autoplay playsinline class="rounded-xl" />
      </div>
    </div>
  </div>
</template>
