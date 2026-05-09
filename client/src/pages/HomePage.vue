<script setup lang="ts">
import { RouterLink } from "vue-router";
import DashboardLayout from "@/components/DashboardLayout.vue";

const btnClass =
  "border rounded-xl m-4 text-2xl p-3 shadow-sm shadow-black text-center hover:scale-105 duration-200 bg-purple-600";
</script>

<template>
  <DashboardLayout>
    <main>
      <header
        class="flex text-clip items-center text-white md:p-0 md:py-20 px-10 md:px-20"
      >
        <h2 class="text-5xl md:mt-10 font-bold">
          <div class="md:grid md:grid-cols-2">
            <span class="animate-fade">
              Turning any camera into a parking monitor.
            </span>
          </div>

          <p class="text-xl md:text-2xl font-normal">
            a browser application that allows you to <br />
            turn your parking situation into a smart one.
          </p>

          <RouterLink to="/vision">
            <button :class="btnClass">Try Parker</button>
          </RouterLink>

          <RouterLink to="/about">
            <button
              class="block sm:hidden my-5 text-2xl duration-300 rounded-lg bg-orangeFadeSides p-4 hover:scale-105"
            >
              About
            </button>
          </RouterLink>
        </h2>

        <div>
          <video
            class="hidden lg:block sm:mx-2 rounded-3xl object-contain animate-fade"
            autoplay
            width="640"
            height="480"
            muted
            loop
            playsinline
            src="/demo.mp4"
          />
        </div>
      </header>
    </main>
  </DashboardLayout>
</template>
