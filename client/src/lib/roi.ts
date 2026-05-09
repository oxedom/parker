import uniqid from "uniqid";
import type { DetectionRoi, RoiCords, SelectedRoi } from "@/types";
import { xywh2xyxy } from "@/lib/geometry";
import { isVehicle, LABELS } from "@/lib/labels";

const MODEL_DIM = 640;

export function selectedFactory(cords: RoiCords): SelectedRoi {
  return {
    label: "vehicle",
    cords: { ...cords },
    uid: `${uniqid()}DATE${Date.now()}`,
    area: Math.round(cords.width * cords.height),
    firstSeen: null,
    lastSeen: null,
    occupied: null,
    evaluating: true,
    hover: false,
    cycleCount: 0,
    events: [
      {
        eventName: "initialized",
        timeMarked: Date.now(),
        duration: null,
      },
    ],
  };
}

export function detectionsToRois(
  detections: number[][],
  imageWidth: number,
  imageHeight: number,
  vehicleOnly: boolean,
): DetectionRoi[] {
  if (!detections?.length) return [];

  const out: DetectionRoi[] = [];
  const sx = imageWidth / MODEL_DIM;
  const sy = imageHeight / MODEL_DIM;

  for (const det of detections) {
    const score = det[4];
    const klass = det[5];
    const label = LABELS[klass];
    if (vehicleOnly && !isVehicle(label)) continue;

    const [x1, y1, x2, y2] = xywh2xyxy(det.slice(0, 4));
    const right_x = x1 * sx;
    const top_y = y1 * sy;
    const width = (x2 - x1) * sx;
    const height = (y2 - y1) * sy;

    out.push({
      cords: {
        right_x,
        top_y,
        left_x: x2 * sx,
        bottom_y: y2 * sy,
        width,
        height,
      },
      label,
      confidenceLevel: score,
      area: width * height,
    });
  }

  return out;
}
