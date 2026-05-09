<script setup lang="ts">
import { useRoute, useRouter, RouterLink } from "vue-router";

const route = useRoute();
const router = useRouter();

function goVision() {
  if (route.path === "/vision") {
    window.location.reload();
  } else {
    router.push("/vision");
  }
}

function navlinkClass(target: string) {
  return route.path === target
    ? "text-white"
    : "text-gray-200 hover:text-white";
}
</script>

<template>
  <div
    class="my-5 sm:my-0 block w-full text-center max-w-[1028px] h-[50px] sm:h-16"
  >
    <nav class="z-[100] absolute">
      <div class="flex gap-2 justify-center items-center">
        <RouterLink to="/">
          <h1
            class="text-bold text-3xl text-center font-bold text-white hover:cursor-pointer"
          >
            <p
              class="p-2 ml-5 uppercase flex justify-items-center items-center gap-5"
            >
              Parker
            </p>
          </h1>
        </RouterLink>

        <div class="flex justify-self-center items-center px-5 gap-5">
          <div
            class="border-2 hidden sm:block px-2 hover:scale-105 duration-200 rounded border-orange-600 cursor-pointer"
            @click="goVision"
          >
            <span
              class="text-2xl font-bold"
              :class="navlinkClass('/vision')"
            >
              Try now
            </span>
          </div>
          <RouterLink to="/about" class="text-2xl font-bold" :class="navlinkClass('/about')">
            About
          </RouterLink>
        </div>
      </div>
    </nav>
  </div>
</template>
