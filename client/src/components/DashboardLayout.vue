<script setup lang="ts">
import NavBar from "@/components/NavBar.vue";
</script>

<template>
  <div
    class="md:h-screen flex flex-col justify-between min-h-screen bg-fixed bg-no-repeat bg-cover bg-hero"
  >
    <NavBar />
    <main class="grow flex flex-col justify-center items-center">
      <slot />
    </main>
  </div>
</template>
