import type { DetectionRoi, SelectedRoi } from "@/types";

type AnyRoi = SelectedRoi | DetectionRoi;

function colorFor(roi: AnyRoi): string {
  if (roi.hover) return "#ffc400";
  if (roi.hover === null) return "#0062CC";
  if (roi.evaluating) return "#808080";
  if (roi.occupied) return "#CC3333";
  return "#22c55e";
}

export function clearCanvas(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
}

export function renderRoi(roi: AnyRoi, ctx: CanvasRenderingContext2D) {
  const { right_x, top_y, width, height } = roi.cords;
  const color = colorFor(roi);

  ctx.lineWidth = roi.hover ? 7 : 4;
  ctx.strokeStyle = color;
  ctx.strokeRect(right_x, top_y, width, height);

  if (roi.label === "vehicle") return;

  ctx.font = "26px Courier";
  ctx.textAlign = "center";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(roi.label, right_x + width / 2, top_y * 0.9);
}

export function renderAllOverlaps(
  overlaps: DetectionRoi[],
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h);
  for (const o of overlaps) {
    o.hover = null;
    renderRoi(o, ctx);
  }
}

export function drawCenteredText(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  text: string,
  yRatio = 0.25,
) {
  ctx.textAlign = "center";
  ctx.fillStyle = "#FF0000";
  ctx.font = "bold 72px Courier";
  ctx.fillText(text, w / 2, h * yRatio);
  ctx.fillStyle = "#000000";
}
