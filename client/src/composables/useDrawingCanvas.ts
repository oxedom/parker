import { onBeforeUnmount, onMounted, ref, watch, type Ref } from "vue";
import type { RoiCords, SelectedRoi } from "@/types";
import { rectangleArea } from "@/lib/geometry";
import { renderRoi } from "@/lib/canvas";

export interface UseDrawingCanvasOptions {
  /** Layer the user draws onto (top). */
  drawCanvas: Ref<HTMLCanvasElement | null>;
  /** Layer that renders all completed ROIs (bottom of the two). */
  overlayCanvas: Ref<HTMLCanvasElement | null>;
  rois: Ref<SelectedRoi[]>;
  imageWidth: Ref<number>;
  imageHeight: Ref<number>;
  onRoiCreated: (cords: RoiCords) => void;
  /** Min area required to register the click as an ROI (px^2). */
  minArea?: number;
}

const SELECTING_COLOR = "#979A9A";
const SELECTED_COLOR = "#f52222";

export function useDrawingCanvas(opts: UseDrawingCanvasOptions) {
  const isDown = ref(false);
  const startX = ref(0);
  const startY = ref(0);
  const offsetX = ref(0);
  const offsetY = ref(0);
  const currentCords = ref<RoiCords>({
    right_x: 0,
    top_y: 0,
    width: 1,
    height: 0,
  });

  let drawCtx: CanvasRenderingContext2D | null = null;
  let overlayCtx: CanvasRenderingContext2D | null = null;

  function updateBounding() {
    const el = opts.drawCanvas.value;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    offsetX.value = rect.left;
    offsetY.value = rect.top;
  }

  function eventToCords(
    px: number,
    py: number,
    w: number,
    h: number,
  ): RoiCords {
    return {
      right_x: w < 0 ? px - Math.abs(w) : px,
      top_y: h < 0 ? py - Math.abs(h) : py,
      width: Math.abs(w),
      height: Math.abs(h),
    };
  }

  function renderAll() {
    if (!overlayCtx || !opts.drawCanvas.value) return;
    const w = opts.drawCanvas.value.width;
    const h = opts.drawCanvas.value.height;
    overlayCtx.clearRect(0, 0, w, h);
    for (const r of opts.rois.value) renderRoi(r, overlayCtx);
    if (!isDown.value && drawCtx) drawCtx.clearRect(0, 0, w, h);
  }

  function onMouseDown(e: MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    startX.value = Math.round(e.clientX - offsetX.value);
    startY.value = Math.round(e.clientY - offsetY.value);
    if (drawCtx) {
      drawCtx.strokeStyle = SELECTING_COLOR;
      drawCtx.lineWidth = 7;
    }
    isDown.value = true;
  }

  function onMouseMove(e: MouseEvent) {
    if (!isDown.value || !drawCtx || !opts.drawCanvas.value) return;
    e.preventDefault();
    e.stopPropagation();

    const mx = Math.round(e.clientX - offsetX.value);
    const my = Math.round(e.clientY - offsetY.value);
    const w = mx - startX.value;
    const h = my - startY.value;

    drawCtx.clearRect(
      0,
      0,
      opts.drawCanvas.value.width,
      opts.drawCanvas.value.height,
    );
    drawCtx.strokeRect(startX.value, startY.value, w, h);

    currentCords.value = eventToCords(startX.value, startY.value, w, h);
  }

  function onMouseUp(e: MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    isDown.value = false;
    const minArea = opts.minArea ?? 500;
    if (rectangleArea(currentCords.value) > minArea) {
      opts.onRoiCreated(currentCords.value);
    }
    currentCords.value = { right_x: 0, top_y: 0, width: 1, height: 0 };
  }

  function onMouseOut(e: MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    isDown.value = false;
  }

  onMounted(() => {
    if (opts.drawCanvas.value) {
      drawCtx = opts.drawCanvas.value.getContext("2d");
      if (drawCtx) {
        drawCtx.strokeStyle = SELECTING_COLOR;
        drawCtx.lineWidth = 7;
      }
    }
    if (opts.overlayCanvas.value) {
      overlayCtx = opts.overlayCanvas.value.getContext("2d");
      if (overlayCtx) {
        overlayCtx.strokeStyle = SELECTED_COLOR;
        overlayCtx.lineWidth = 7;
      }
    }
    updateBounding();
    window.addEventListener("resize", updateBounding);
    window.addEventListener("scroll", updateBounding, true);
  });

  onBeforeUnmount(() => {
    window.removeEventListener("resize", updateBounding);
    window.removeEventListener("scroll", updateBounding, true);
  });

  watch(
    [opts.rois, opts.imageWidth, opts.imageHeight],
    () => {
      updateBounding();
      renderAll();
    },
    { deep: true },
  );

  return {
    isDown,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onMouseOut,
  };
}
