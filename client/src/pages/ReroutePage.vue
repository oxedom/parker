<script setup lang="ts">
import { useRoute, useRouter } from "vue-router";

const route = useRoute();
const router = useRouter();

function go(target: "input" | "view") {
  const remoteID = (route.query.remoteID as string) ?? "";
  router.push({ path: `/${target}`, query: { remoteID } });
}
</script>

<template>
  <div
    class="h-screen gap-2 pt-10 flex flex-col min-h-screen bg-fixed bg-no-repeat bg-cover bg-filler w-full grow items-center"
  >
    <div class="flex flex-col m-5 justify-center font-bold gap-2">
      <header>
        <h2 class="text-center text-white text-6xl">Menu</h2>
      </header>
      <button
        class="bg-purple-500 rounded-md text-white text-3xl p-5"
        @click="go('input')"
      >
        Stream Video
      </button>
      <button
        class="bg-orange-500 rounded-md text-white text-3xl p-5"
        @click="go('view')"
      >
        View Parking Situation
      </button>
    </div>
  </div>
</template>
