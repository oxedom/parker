<script setup lang="ts">
import AppButton from "@/components/AppButton.vue";

defineProps<{ isOpen: boolean }>();
const emit = defineEmits<{ close: [] }>();
</script>

<template>
  <Teleport to="body">
    <div
      v-if="isOpen"
      class="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-[99999]"
      @click.self="emit('close')"
    >
      <div class="bg-filler p-3 rounded-xl">
        <slot />
        <AppButton intent="destructive" class="mt-2" @click="emit('close')">
          <span class="text-lg">Close</span>
        </AppButton>
      </div>
    </div>
  </Teleport>
</template>
