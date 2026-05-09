<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import type { SelectedRoi } from "@/types";
import { totalOccupied } from "@/lib/geometry";
import { usePeer } from "@/composables/usePeer";

const route = useRoute();
const router = useRouter();
const peer = usePeer();

const counts = ref<{ OccupiedCount: number | string; availableCount: number | string }>({
  OccupiedCount: "...",
  availableCount: "...",
});

onMounted(async () => {
  await peer.init();
  const remoteId = (route.query.remoteID as string) ?? "";
  if (!remoteId) return;

  const conn = peer.connectTo(remoteId);
  conn?.on("data", (data: unknown) => {
    if (Array.isArray(data)) {
      counts.value = totalOccupied(data as SelectedRoi[]);
    }
  });
});
</script>

<template>
  <div>
    <nav
      class="flex border-b-2 border-black justify-around items-center h-[80px] bg-filler text-white"
    >
      <h4
        class="text-4xl font-bold text-center uppercase hover:cursor-pointer"
        @click="router.push('/')"
      >
        Parker
      </h4>
      <span
        class="text-xl font-bold text-center hover:cursor-pointer"
        @click="router.push('/about')"
      >
        About
      </span>
    </nav>
    <div
      class="flex flex-col items-center w-full h-screen min-h-screen bg-fixed bg-no-repeat bg-cover bg-filler grow"
    >
      <div class="grid w-full h-full grid-rows-2 md:grid-rows-none md:grid-cols-2">
        <div
          class="flex flex-col items-center justify-center w-full gap-2 bg-green-700"
        >
          <p class="text-6xl font-bold text-white rounded">Available</p>
          <span class="text-5xl text-white">{{ counts.availableCount }}</span>
        </div>
        <div
          class="flex flex-col items-center justify-center w-full gap-2 bg-red-700"
        >
          <p class="text-6xl font-bold text-white border-red-700 rounded">
            Occupied
          </p>
          <span class="text-5xl text-white">{{ counts.OccupiedCount }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
