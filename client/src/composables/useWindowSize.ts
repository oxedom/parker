import { onBeforeUnmount, onMounted, ref } from "vue";

export function useWindowSize() {
  const width = ref(0);
  const height = ref(0);

  function update() {
    width.value = window.innerWidth;
    height.value = window.innerHeight;
  }

  onMounted(() => {
    update();
    window.addEventListener("resize", update);
  });
  onBeforeUnmount(() => window.removeEventListener("resize", update));

  return { width, height };
}
