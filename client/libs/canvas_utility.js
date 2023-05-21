import labels from "./labels.json";

export function drawCanvas(canvasEl, img) {
  canvasEl.width = getComputedStyle(canvasEl).width.split("px")[0];
  canvasEl.height = getComputedStyle(canvasEl).height.split("px")[0];

  let ratio = Math.min(
    canvasEl.width / img.width,
    canvasEl.height / img.height
  );
  let x = (canvasEl.width - img.width * ratio) / 2;
  let y = (canvasEl.height - img.height * ratio) / 2;

  canvasEl.getContext("2d").clearRect(0, 0, canvasEl.width, canvasEl.height);
  canvasEl
    .getContext("2d")
    .drawImage(
      img,
      0,
      0,
      img.width,
      img.height,
      x,
      y,
      img.width * ratio,
      img.height * ratio
    );
}

export async function imageCapturedToCanvas(capturedImage, inputCanvasRef) {
  try {
    const blob = await capturedImage.takePhoto();
    const imageBitmap = await createImageBitmap(blob);
    drawCanvas(inputCanvasRef.current, imageBitmap);
  } catch (error) {
    console.log(error);
  }
}

export function drawTextOnCanvas(canvas, width, height, text) {
  const font = "bold 72px Courier ";
  canvas.current.textAlign = "center";
  canvas.current.fillStyle = "#FF0000";
  canvas.current.font = font;
  canvas.current.fillText(text, width / 2, height / 4);
  canvas.current.fillStyle = "#000000";
}

export function rectangleArea(rect) {
  return Math.abs(rect.width * rect.height);
}

function setColor(roi) {
  let color = "#22c55e";

  if (roi.occupied && roi.hover == false) {
    //bg-red-500 hex
    color = "#CC3333";
  }

  if (roi.evaluating) {
    color = "#808080";
  }

  if (roi.hover === null) {
    color = "#0062CC";
  }

  if (roi.hover) {
    color = "#ffc400";
  }
  return color;
}

export function renderRoi(roi, contextCanvas) {
  //Cords
  const { height, right_x, top_y, width } = roi.cords;
  const context = contextCanvas.current;
  let borderWidth = 5;
  const offset = borderWidth * 2;
  context.lineWidth = 4;
  //bg-green-500 hex
  let color = setColor(roi);
  context.strokeStyle = color;

  if (roi.hover) {
    context.lineWidth = 7;
  }

  //Gets centerX
  const centerX = right_x + width / 2;
  //Font and Size needs to be state

  //Draws a rect on the detection
  context.strokeStyle = color;

  contextCanvas.current.strokeRect(right_x, top_y, width, height);

  const font = "26px Courier";
  contextCanvas.current.textAlign = "center";
  contextCanvas.current.font = font;
  contextCanvas.textBaseline = "top";
  contextCanvas.fillStyle = "#ffffff";
  //This prevents the canvas to draw the user selected roi,
  //because the label vechicle does not exist in the yolo7 labels (too generic)
  if (roi.label === "vehicle") {
    return;
  }
  contextCanvas.current.fillText(roi.label, centerX, top_y * 0.9);
}

export function renderAllOverlaps(overlaps, canvasRef, width, height) {
  //Clears canvas before rendering all overlays (Runs each response)
  //For each on the detections
  canvasRef.current.clearRect(0, 0, width, height);
  overlaps.forEach((o) => {
    o.hover = null;

    renderRoi(o, canvasRef);
  });
}

export function clearCanvas(canvasRef, width, height) {
  if (canvasRef.current != null) {
    canvasRef.current.clearRect(0, 0, width, height);
  }
}

export function xywh2xyxy(x) {
  //Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  var y = [];
  y[0] = x[0] - x[2] / 2; //top left x
  y[1] = x[1] - x[3] / 2; //top left y
  y[2] = x[0] + x[2] / 2; //bottom right x
  y[3] = x[1] + x[3] / 2; //bottom right y
  return y;
}

export const renderBoxes = (
  canvasRef,
  threshold,
  boxes_data,
  scores_data,
  classes_data
) => {
  canvasRef.clearRect(0, 0, canvasRef.canvas.width, canvasRef.canvas.height); // clean canvas

  // font configs
  const font = "18px sans-serif";
  canvasRef.font = font;
  canvasRef.textBaseline = "top";

  for (let i = 0; i < scores_data.length; ++i) {
    if (scores_data[i] > threshold) {
      const klass = labels[classes_data[i]];
      const score = (scores_data[i] * 100).toFixed(1);

      let [x1, y1, x2, y2] = xywh2xyxy(boxes_data[i]);

      const width = x2 - x1;
      const height = y2 - y1;

      // Draw the bounding box.
      canvasRef.strokeStyle = "#B033FF";
      canvasRef.lineWidth = 2;
      canvasRef.strokeRect(x1, y1, width, height);

      // Draw the label background.
      canvasRef.fillStyle = "#B033FF";
      const textWidth = canvasRef.measureText(
        klass + " - " + score + "%"
      ).width;
      const textHeight = parseInt(font, 10); // base 10
      canvasRef.fillRect(
        x1 - 1,
        y1 - (textHeight + 2),
        textWidth + 2,
        textHeight + 2
      );

      // Draw labels
      canvasRef.fillStyle = "#ffffff";
      canvasRef.fillText(
        klass + " - " + score + "%",
        x1 - 1,
        y1 - (textHeight + 2)
      );
    }
  }
};
