<script setup lang="ts">
defineProps<{
  modelValue: boolean;
  text: string;
}>();

const emit = defineEmits<{
  "update:modelValue": [boolean];
}>();

function onChange(e: Event) {
  emit("update:modelValue", (e.target as HTMLInputElement).checked);
}
</script>

<template>
  <div class="flex flex-col">
    <span class="pb-1 mr-3 font-bold text-left text-white drop-shadow-sm">{{
      text
    }}</span>
    <label class="relative flex items-center cursor-pointer select-none w-max">
      <input
        type="checkbox"
        :checked="modelValue"
        class="transition-colors bg-red-500 rounded-full appearance-none cursor-pointer checked:bg-green-500 w-14 h-7 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black focus:ring-blue-500"
        @change="onChange"
      />
      <span
        class="w-5 h-5 right-8 absolute rounded-full transform transition-transform bg-gray-200 pointer-events-none"
        :class="{ 'translate-x-7': modelValue }"
      />
    </label>
  </div>
</template>
