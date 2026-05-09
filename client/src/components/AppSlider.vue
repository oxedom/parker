<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    modelValue: number;
    label: string;
    min?: number;
    max?: number;
    step?: number;
    unit?: string;
  }>(),
  { min: 10, max: 100, step: 1, unit: "%" },
);

const emit = defineEmits<{
  "update:modelValue": [number];
  change: [number];
}>();

function onInput(e: Event) {
  const v = Number((e.target as HTMLInputElement).value);
  emit("update:modelValue", v);
  emit("change", v);
}
</script>

<template>
  <div class="flex flex-col justify-center text-white">
    <label class="font-bold text-left drop-shadow-sm">{{ label }}</label>
    <div class="grid grid-cols-[auto_1fr] gap-2 whitespace-nowrap">
      <span class="text-left">{{ modelValue }}{{ unit }}</span>
      <input
        type="range"
        :min="min"
        :max="max"
        :step="step"
        class="w-full mr-4 accent-blue-400"
        :value="modelValue"
        @input="onInput"
      />
    </div>
  </div>
</template>
