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
  //Gets centerX
  const centerX = right_x + width / 2;
  //Font and Size needs to be state
  contextCanvas.current.font = "72px Courier";
  contextCanvas.current.textAlign = "center";
  //Draws a rect on the detection
  context.strokeStyle = roi.color;
  context.lineWidth = 10;
  contextCanvas.current.strokeRect(right_x, top_y, width, height);

  context.lineWidth = 11;
  context.strokeStyle = "#000000";
  contextCanvas.current.strokeRect(right_x - 8, top_y, width, height);

  contextCanvas.current.fillText(roi.label, centerX, top_y * 0.8);
}
