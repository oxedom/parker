import { defineStore } from "pinia";
import { ref } from "vue";

export const useDimensionsStore = defineStore("dimensions", () => {
  const imageWidth = ref(640);
  const imageHeight = ref(480);

  function set(width: number, height: number) {
    imageWidth.value = width;
    imageHeight.value = height;
  }

  return { imageWidth, imageHeight, set };
});
