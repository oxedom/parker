import { defineStore } from "pinia";
import { computed, ref } from "vue";
import type { DetectionRoi, RoiCords, SelectedRoi } from "@/types";
import { selectedFactory } from "@/lib/roi";
import { checkRectOverlap, getOverlap } from "@/lib/geometry";
import { useSettingsStore } from "@/stores/settings";

interface OccupationUpdate {
  predictions: DetectionRoi[];
  /** ms since epoch */
  now?: number;
}

/**
 * Auto-detect: collect detections over a window, then keep only candidates
 * that appeared with sufficient frequency and consistent position.
 */
function suppressAutoDetections(
  matrix: DetectionRoi[][],
  minimumAttendance: number,
): DetectionRoi[] {
  if (matrix.length === 0) return [];
  let longest = matrix[0];
  for (const row of matrix) if (row.length > longest.length) longest = row;

  const scores = new Array<number>(longest.length).fill(0);
  for (let i = 0; i < longest.length; i++) {
    const cur = longest[i];
    for (const row of matrix) {
      for (const det of row) {
        const overlap = getOverlap(cur.cords, det.cords);
        if (!overlap) continue;
        const overlapArea = Math.round(overlap.area);
        const curArea = Math.round(cur.area);
        const ratio = overlapArea / curArea;
        if (ratio > 0.8 || overlapArea === curArea) scores[i]++;
      }
    }
  }

  const threshold = Math.ceil(matrix.length * minimumAttendance);
  return longest.filter((_, i) => scores[i] > threshold);
}

export const useRoisStore = defineStore("rois", () => {
  const settings = useSettingsStore();

  const items = ref<SelectedRoi[]>([]);

  // Auto-detect transient state
  const autoDetectArr = ref<DetectionRoi[][]>([]);
  const autoCheckedAt = ref(0);

  const counts = computed(() => {
    let occupied = 0;
    let available = 0;
    for (const r of items.value) {
      if (r.occupied === true) occupied++;
      else if (r.occupied === false) available++;
    }
    return { occupied, available };
  });

  function add(cords: RoiCords) {
    items.value = [...items.value, selectedFactory(cords)];
  }

  function remove(uid: string) {
    items.value = items.value.filter((r) => r.uid !== uid);
  }

  function clear() {
    items.value = [];
  }

  function setHover(uid: string, hover: boolean) {
    items.value = items.value.map((r) =>
      r.uid === uid ? { ...r, hover } : r,
    );
  }

  function importFromJson(payload: { selectedRegions: SelectedRoi[] }) {
    items.value = payload.selectedRegions ?? [];
  }

  function startAutoDetect() {
    settings.autoDetect = true;
    autoDetectArr.value = [];
    autoCheckedAt.value = 0;
  }

  function processAutoDetectFrame(predictions: DetectionRoi[]) {
    const now = Date.now();
    if (autoCheckedAt.value === 0) {
      autoCheckedAt.value = now;
      return;
    }

    const elapsed = now - autoCheckedAt.value;
    if (elapsed <= settings.autoEvaluateTime) {
      autoDetectArr.value = [...autoDetectArr.value, predictions];
      return;
    }

    const survivors = suppressAutoDetections(autoDetectArr.value, 0.6);
    items.value = survivors.map((s) => selectedFactory(s.cords));
    settings.autoDetect = false;
    autoDetectArr.value = [];
    autoCheckedAt.value = 0;
  }

  /**
   * Run a single occupation update from a fresh batch of detections.
   * The update mutates the existing rois with new firstSeen/lastSeen/occupied data.
   */
  function processFrame({ predictions, now = Date.now() }: OccupationUpdate) {
    if (settings.autoDetect) {
      processAutoDetectFrame(predictions);
      return;
    }

    if (items.value.length === 0) return;

    // Use a sentinel "no overlap" detection so the loop runs and resets state.
    const detections: DetectionRoi[] =
      predictions.length > 0
        ? predictions
        : [
            {
              cords: { right_x: -999, top_y: -999, width: -999, height: -999 },
              label: "car",
              confidenceLevel: 99,
              area: -999,
            },
          ];

    const evaluateTime = settings.evaluateTime;
    const overlapThreshold = settings.overlapThreshold;

    items.value = items.value.map((roi) => {
      const next: SelectedRoi = { ...roi, events: [...roi.events] };
      const isOverlap = checkRectOverlap(next, detections, overlapThreshold);

      if (next.evaluating) {
        const stillEvaluating =
          now - (next.events[0]?.timeMarked ?? now) < evaluateTime;
        if (!stillEvaluating) next.evaluating = false;
      }

      const firstDetect = isOverlap && next.firstSeen === null;
      const overlapsKnown = isOverlap && next.firstSeen !== null;

      if (firstDetect) {
        next.firstSeen = now;
        next.lastSeen = now;
      } else if (overlapsKnown) {
        next.lastSeen = now;
        const timeDiff = (next.lastSeen ?? now) - (next.firstSeen ?? now);

        if (timeDiff > evaluateTime) {
          if (!next.occupied) {
            next.occupied = true;
            next.cycleCount += 1;
            next.events.push({
              cycle: next.cycleCount,
              eventName: "occupied",
              timeMarked: now,
              duration: null,
            });
          }
          const idx = next.cycleCount;
          if (next.events[idx]) {
            next.events[idx].duration =
              (next.lastSeen ?? 0) - (next.firstSeen ?? 0);
          }
        }
      } else if (now - (next.lastSeen ?? 0) > evaluateTime) {
        if (next.occupied) {
          next.firstSeen = null;
          next.lastSeen = null;
          next.occupied = false;
        }
      }

      return next;
    });
  }

  return {
    items,
    counts,
    add,
    remove,
    clear,
    setHover,
    importFromJson,
    startAutoDetect,
    processFrame,
  };
});
