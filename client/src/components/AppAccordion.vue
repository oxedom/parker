<script setup lang="ts">
import { ref } from "vue";

withDefaults(
  defineProps<{
    title: string;
    initialOpen?: boolean;
  }>(),
  { initialOpen: true },
);

const isOpen = ref(true);
function toggle() {
  isOpen.value = !isOpen.value;
}
</script>

<template>
  <div class="w-full px-3 mx-auto">
    <div
      class="flex duration-200 justify-between py-2 mb-2 items-center cursor-pointer"
      :class="{ 'border-b border-gray-200': isOpen }"
      @click="toggle"
    >
      <h3 class="text-2xl font-medium text-white">{{ title }}</h3>
      <svg
        class="w-6 h-6 transition-transform invert duration-300 transform"
        :class="{ 'rotate-180': isOpen }"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <polyline points="6 9 12 15 18 9" />
      </svg>
    </div>
    <div v-show="isOpen" class="duration-75 animate-fade">
      <div class="flex flex-col gap-2 text-center">
        <slot />
      </div>
    </div>
  </div>
</template>
