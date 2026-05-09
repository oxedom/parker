import { onUnmounted, ref, type Ref } from "vue";
import { detectWebcam, getDefaultVideoSettings } from "@/lib/webrtc";

export interface UseWebcamOptions {
  videoEl: Ref<HTMLVideoElement | null>;
}

export function useWebcam({ videoEl }: UseWebcamOptions) {
  const stream = ref<MediaStream | null>(null);
  const ready = ref(false);
  const width = ref(640);
  const height = ref(480);
  const error = ref<unknown>(null);

  async function start() {
    try {
      const present = await detectWebcam();
      if (!present) {
        error.value = new Error("No webcam found");
        return;
      }

      const dims = await getDefaultVideoSettings();
      width.value = dims.width;
      height.value = dims.height;

      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: dims.width }, height: { ideal: dims.height } },
      });
      stream.value = s;

      if (videoEl.value) {
        videoEl.value.srcObject = s;
        await videoEl.value.play().catch(() => {});
      }
      ready.value = true;
    } catch (e) {
      error.value = e;
      console.error(e);
    }
  }

  function stop() {
    stream.value?.getTracks().forEach((t) => t.stop());
    stream.value = null;
    ready.value = false;
    if (videoEl.value) videoEl.value.srcObject = null;
  }

  onUnmounted(stop);

  return { stream, ready, width, height, error, start, stop };
}
