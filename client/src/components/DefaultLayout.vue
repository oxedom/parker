<script setup lang="ts">
import NavBar from "@/components/NavBar.vue";
</script>

<template>
  <div class="h-screen flex flex-col justify-between min-h-screen">
    <nav class="bg-heroShort bg-no-repeat bg-cover">
      <NavBar />
    </nav>
    <main class="bg-slate-100 grow">
      <slot />
    </main>
  </div>
</template>
