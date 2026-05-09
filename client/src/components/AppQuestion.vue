<script setup lang="ts">
import { ref } from "vue";

defineProps<{
  question: string;
  answer: string;
}>();

const isOpen = ref(false);
</script>

<template>
  <div class="border border-gray-200 p-4 rounded-md mb-4">
    <button
      class="w-full text-left font-medium mb-2 focus:outline-none"
      @click="isOpen = !isOpen"
    >
      <span class="text-black text-2xl">{{ question }}</span>
    </button>
    <div
      class="overflow-hidden transition-all duration-300"
      :class="isOpen ? 'h-auto' : 'h-0'"
    >
      <div class="text-gray-900">
        <p>{{ answer }}</p>
      </div>
    </div>
  </div>
</template>
