<script setup lang="ts">
import { useRoisStore } from "@/stores/rois";
import AppButton from "@/components/AppButton.vue";

const rois = useRoisStore();

function handleExportJson() {
  const payload = {
    title: window.location.href,
    repo: "https://github.com/oxedom/parker",
    timeOfExport: Date.now(),
    rois: rois.items,
  };
  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "exported_data.json";
  link.click();
  URL.revokeObjectURL(url);
}
</script>

<template>
  <div
    class="bg-black/60 backdrop-blur-sm pb-3 px-5 flex flex-col max-w-[500px]"
  >
    <h2
      class="text-center text-3xl text-white pt-2 border-b border-orange-500 pb-1"
    >
      Data manager
    </h2>
    <p class="text-white text-xl pt-4">
      Selection boxes keep track of events that occur. Export that data locally
      to perform your own data exploration.
    </p>
    <div class="flex flex-col items-center mt-10 gap-y-2">
      <h1 class="text-white text-2xl">Export as</h1>
      <div class="flex justify-center items-center gap-5">
        <AppButton @click="handleExportJson">
          <span class="text-lg">JSON</span>
        </AppButton>
      </div>
    </div>
  </div>
</template>
