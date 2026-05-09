<script setup lang="ts">
import { computed } from "vue";

type Intent = "primary" | "secondary" | "destructive";
type Size = "sm" | "md" | "lg";

const props = withDefaults(
  defineProps<{
    intent?: Intent;
    size?: Size;
    fullWidth?: boolean;
    disabled?: boolean;
  }>(),
  { intent: "secondary", size: "md", fullWidth: false, disabled: false },
);

defineEmits<{ click: [MouseEvent] }>();

const colors: Record<Intent, string> = {
  primary: "bg-blue-500 hover:bg-blue-400 text-white",
  secondary: "bg-white hover:bg-gray-200 text-slate-800",
  destructive: "bg-red-500 hover:bg-red-400 text-white",
};
const sizes: Record<Size, string> = {
  sm: "text-sm",
  md: "text-base",
  lg: "text-lg",
};

const classes = computed(
  () =>
    `py-1.5 px-3 font-medium rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
      colors[props.intent]
    } ${sizes[props.size]} ${props.fullWidth ? "w-full" : ""}`,
);
</script>

<template>
  <button :class="classes" :disabled="disabled" @click="$emit('click', $event)">
    <slot />
  </button>
</template>
