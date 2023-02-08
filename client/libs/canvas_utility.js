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

export function rectangleArea(rect) {
  return Math.abs(rect.width * rect.height);
}

export function renderRoi(roi, contextCanvas) {
  //Cords
  const { height, right_x, top_y, width } = roi.cords;
  const context = contextCanvas.current;
  const borderWidth = 5;
  const offset = borderWidth * 2;
  //bg-green-500 hex
  let color = "#22c55e";


  if (roi.occupied && roi.hover == false) {
    //bg-red-500 hex
    color = "#CC3333";
  }
  if (Date.now() - roi.time < 6000) {
    color = "#808080";
  }
  
  if (roi.hover) {
    color = "#FF0000";
  }
  //Gets centerX
  const centerX = right_x + width / 2;
  //Font and Size needs to be state
  contextCanvas.current.font = "72px Courier";
  contextCanvas.current.textAlign = "center";
  //Draws a rect on the detection
  context.strokeStyle = color;
  context.lineWidth = 10;

  contextCanvas.current.strokeRect(right_x, top_y, width, height);
  context.strokeStyle = "#333333";
  context.lineWidth = 1.5;
  contextCanvas.current.strokeRect(
    right_x - borderWidth,
    top_y - borderWidth,
    width + offset,
    height + offset
  );

  // contextCanvas.current.fillText(roi.label, centerX, top_y * 0.8);
}

export function renderAllOverlaps(overlaps, canvasRef, width, height) {
  //Clears canvas before rendering all overlays (Runs each response)
  //For each on the detections
  canvasRef.current.clearRect(0, 0, width, height);
  overlaps.forEach((o) => {
    o.hover = true;
    renderRoi(o, canvasRef);
  });
}

export function clearCanvas(canvasRef, width, height) {
  if (canvasRef.current != null) {
    canvasRef.current.clearRect(0, 0, width, height);
  }
}
