import type { RoiCords, DetectionRoi, SelectedRoi } from "@/types";

export function rectangleArea(rect: { width: number; height: number }): number {
  return Math.abs(rect.width * rect.height);
}

export function xywh2xyxy(x: number[]): [number, number, number, number] {
  return [x[0] - x[2] / 2, x[1] - x[3] / 2, x[0] + x[2] / 2, x[1] + x[3] / 2];
}

export interface OverlapResult extends RoiCords {
  area: number;
}

export function getOverlap(a: RoiCords, b: RoiCords): OverlapResult | null {
  const x1 = Math.max(a.right_x, b.right_x);
  const x2 = Math.min(a.right_x + a.width, b.right_x + b.width);
  if (x2 < x1) return null;

  const y1 = Math.max(a.top_y, b.top_y);
  const y2 = Math.min(a.top_y + a.height, b.top_y + b.height);
  if (y2 < y1) return null;

  return {
    right_x: x1,
    top_y: y1,
    width: x2 - x1,
    height: y2 - y1,
    area: (x2 - x1) * (y2 - y1),
  };
}

export function checkRectOverlap(
  rect: SelectedRoi,
  detections: DetectionRoi[],
  overlapThreshold: number,
): boolean {
  for (const d of detections) {
    const overlap = getOverlap(d.cords, rect.cords);
    if (!overlap) continue;

    const overlapArea = Math.round(overlap.area);
    const rectArea = Math.round(rect.area);

    if (overlapArea === rectArea) return true;
    if (overlapArea / rectArea > overlapThreshold) return true;
  }
  return false;
}

export function totalOccupied(rois: SelectedRoi[]): {
  OccupiedCount: number;
  availableCount: number;
} {
  let OccupiedCount = 0;
  let availableCount = 0;
  for (const roi of rois) {
    if (roi.occupied === true) OccupiedCount++;
    else if (roi.occupied === false) availableCount++;
  }
  return { OccupiedCount, availableCount };
}
