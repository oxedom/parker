// import { serverRectangleParse, bufferToServer} from "./libs"
const videoInputRef = document.getElementById("webcam");
const overlayRef = document.getElementById("overlay");
const canvasRef = document.getElementById("canvas");
const inputCanvasRef = document.getElementById("input-canvas");
const outputRef = document.getElementById("output");
const image_width = 1280
const image_height = 720
// get references to the canvas and context
async function capturedImageToBuffer(capturedImage)
{
  const imagePhoto = await capturedImage.takePhoto();
  let imageBuffer = await imagePhoto.arrayBuffer();

  imageBuffer = new Uint8Array(imageBuffer);
  return imageBuffer;
}

async function bufferToServer(capturedImage) {
  
  const imageBuffer = await capturedImageToBuffer(capturedImage)

  const res = await fetch(flask_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ buffer: [...imageBuffer] }),
  });

  const data = await res.json();

 
  return data;
}

function renderRectangleFactory(canvasEl,overlayEl) {
  const selectedRegions = [];
  const ctx = canvasEl.getContext("2d");
  const ctxo = overlayEl.getContext("2d");

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
    ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);

    // draw a new rect from the start position
    // to the current mouse position
    ctx.strokeRect(startX, startY, width, height);

    prevStartX = startX;
    prevStartY = startY;

    prevWidth = width;
    prevHeight = height;
  }

  function _addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight) {


    let right_x = null
    let top_y = null;

    prevWidth < 0 ? right_x = prevStartX-Math.abs(prevWidth) : right_x = prevStartX
    prevHeight < 0 ? top_y = prevStartY-Math.abs(prevHeight) : top_y = prevStartY
    
    const  roiObj = {
      height: Math.abs(prevHeight),
      right_x: right_x,
      top_y: top_y,
      width: Math.abs(prevWidth),
     
    }



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

const renderRectangle = renderRectangleFactory(canvasRef, overlayRef);

canvasRef.addEventListener("mousedown", (e) => {
  e.preventDefault();
  renderRectangle.handleMouseDown(e);
});

canvasRef.addEventListener("mousemove", (e) => {
  e.preventDefault();
  renderRectangle.handleMouseMove(e);
});

canvasRef.addEventListener("mouseout", (e) => {
  e.preventDefault();
  renderRectangle.handleMouseOut(e);
});

canvasRef.addEventListener("mouseup", (e) => {
  e.preventDefault();
  renderRectangle.handleMouseUp(e);
});

function rectangleArea(rect) {
  const width = rect.cords.right_x - rect.cords.left_x;
  const height = rect.cords.bottom_y - rect.cords.top_y;
  return Math.abs(width * height);
}

let selectedType = "anything";
const flask_url = "http://127.0.0.1:5000/api/cv/yolo";

//State:
const arrOfScreens = [videoInputRef, overlayRef, canvasRef, inputCanvasRef];

//Image input and returns a canvas version of the image (USED so I can draw on the image)
function drawCanvas(canvasEl, img) {
  canvasEl.width = getComputedStyle(canvasEl).width.split("px")[0];
  canvasEl.height = getComputedStyle(canvasEl).height.split("px")[0];
  let ratio = Math.min(canvasEl.width / img.width, canvasEl.height / img.height);
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

const onTakePhotoButtonClick = async (capturedImage) => {
  try {
    const blob = await capturedImage.takePhoto();
    const imageBitmap = await createImageBitmap(blob);
    drawCanvas(inputCanvasRef, imageBitmap);
  } catch (error) {
    console.log(error);
  }
};

//Sets the size of all the canvas to be the same
const setSize = (width, height) => {
  arrOfScreens.forEach((screen) => {
    screen.width = width;
    screen.height = height;
  });
};

function renderVideo(data, imageCaptured) {
  onTakePhotoButtonClick(imageCaptured);
  outputRef.src = data.img;
}

function rectangleArea(rect) {
  
  const width = rect.right_x - rect.left_x;
  const height = rect.bottom_y - rect.top_y;
  return Math.abs(width * height);
}



async function intervalProcessing(track) {
  //Converts imageCaptured parameter to buffer and sends it to the server for computer vision
  //processing and returns an object with a new image with meta_data after processing
  const imageCaptured = new ImageCapture(track);
  const data = await bufferToServer(imageCaptured);
  console.table(data.meta_data.detections.forEach(d => 
    {
      let ctx1 = canvasRef.getContext('2d')
          // to the current mouse position
      const {left_x, top_y, right_x, bottom_y} = d.cords
      console.log(d.label);
      console.table(d.cords);
      ctx1.strokeRect(left_x,Math.max(top_y,bottom_y),right_x-left_x,top_y-bottom_y);
      // console.table(d.cords);
    }));
    //Updates the SRCs and Canvas in order to display Client Server Images


    renderVideo(data, imageCaptured);

  // canvasRef.getContext('2d').strokeStyle = "#66B0E6";
  // canvasRef.getContext('2d').lineWidth = 10;
  // canvasRef.getContext('2d').strokeRect(98,50,171,542)
}

const getVideo = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { min: image_width } },
  });

  //Gets the current screen
  const track = stream.getVideoTracks()[0];
  let { width, height } = track.getSettings();
  setSize(width, height);

  setTimeout (() => {
    intervalProcessing(track);
  }, 1);
};

getVideo();
