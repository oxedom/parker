export function renderRectangleFactory(canvasEl, overlayEl) {
  const selectedRegions = [];
  const ctx = canvasEl.getContext("2d");
  const ctxo = overlayEl.getContext("2d");
  console.log(canvasEl);
  ctx.strokeStyle = "green";
  ctx.lineWidth = 10;
  ctxo.strokeStyle = "blue";
  ctxo.lineWidth = 10;
  let canvasOffset = canvasEl.getBoundingClientRect();
  let offsetX = canvasOffset.left;
  let offsetY = canvasOffset.top;
  // this flage is true when the user is dragging the mouse
  let isDown = false;
  // these vars will hold the starting mouse position
  let startX;
  let startY;

  let prevStartX = 0;
  let prevStartY = 0;

  let prevWidth = 0;
  let prevHeight = 0;

  function handleMouseDown(e) {
    e.preventDefault();
    e.stopPropagation();

    //(0,0) Would be the top left cornor
    //(Max Width, Max Height ) woul be the bottom right cornor
    // save the starting x/y of the rectangle
    startX = parseInt(e.clientX - offsetX);
    startY = parseInt(e.clientY - offsetY);

    // set a flag indicating the drag has begun
    isDown = true;
  }

  function handleMouseUp(e) {
    e.preventDefault();
    e.stopPropagation();

    // the drag is over, clear the dragging flag
    isDown = false;
    // ctxo.strokeRect(random.left_x, random.top_y, random.width, random.height);
    ctx.strokeStyle = "#ABDAFC";
    ctx.lineWidth = 10;
    ctxo.strokeStyle = "#66B0E6";
    ctxo.lineWidth = 10;
    ctxo.strokeRect(prevStartX, prevStartY, prevWidth, prevHeight);

    _addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight);
  }

  function handleMouseOut(e) {
    e.preventDefault();
    e.stopPropagation();

    // the drag is over, clear the dragging flag
    isDown = false;
  }

  function handleMouseMove(e) {
    e.preventDefault();
    e.stopPropagation();

    // if we're not dragging, just return
    if (!isDown) {
      return;
    }

    // get the current mouse position
    //(0,0) Would be the top left cornor
    //(Max Width, Max Height ) woul be the bottom right cornor
    // save the starting x/y of the rectangle
    const mouseX = parseInt(e.clientX - offsetX);
    const mouseY = parseInt(e.clientY - offsetY);

    // calculate the rectangle width/height based
    // on starting vs current mouse position
    var width = mouseX - startX;
    var height = mouseY - startY;

    // clear the canvas
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    // draw a new rect from the start position
    // to the current mouse position
    ctx.strokeRect(startX, startY, width, height);

    prevStartX = startX;
    prevStartY = startY;

    prevWidth = width;
    prevHeight = height;
  }

  function _addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight) {
    let right_x = null;
    let top_y = null;

    prevWidth < 0
      ? (right_x = prevStartX - Math.abs(prevWidth))
      : (right_x = prevStartX);
    prevHeight < 0
      ? (top_y = prevStartY - Math.abs(prevHeight))
      : (top_y = prevStartY);

    const roiObj = {
      height: Math.abs(prevHeight),
      right_x: right_x,
      top_y: top_y,
      width: Math.abs(prevWidth),
    };

    console.log(roiObj);
    selectedRegions.push(roiObj);

    return selectedRegions;
  }

  function getSelectedRegions() {
    return selectedRegions;
  }

  return {
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleMouseOut,
    getSelectedRegions,
  };
}

export function drawCanvas(canvasEl, img) {
  // if(canvasEl === null) { return}

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

export async function onTakePhotoButtonClick(capturedImage, inputCanvasRef) {
  try {
    const blob = await capturedImage.takePhoto();
    const imageBitmap = await createImageBitmap(blob);
    drawCanvas(inputCanvasRef.current, imageBitmap);
  } catch (error) {
    console.log(error);
  }
}

export const setSize = (width, height, screen) => {
  screen.current.width = width;
  screen.current.height = height;
};

export function renderVideo(outputRef, data, imageCaptured) {
  // onTakePhotoButtonClick(imageCaptured);
  outputRef.current.src = data.img;
}
