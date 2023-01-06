// import { serverRectangleParse, bufferToServer} from "./libs"
const videoInputRef = document.getElementById("webcam");
const overlayRef = document.getElementById("overlay");
const canvasRef = document.getElementById("canvas");
const inputCanvasRef = document.getElementById("input-canvas");
const output = document.getElementById("output");


// get references to the canvas and context



async function bufferToServer(capturedImage) {
  const imagePhoto = await capturedImage.takePhoto();
  let imageBuffer = await imagePhoto.arrayBuffer();
  imageBuffer = new Uint8Array(imageBuffer);

  const res = await fetch(flask_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ buffer: [...imageBuffer] }),
  });

  const data = await res.json();

  return data;
}

function renderRectangleFactory() {

  const selectedRegions = [];
  const ctx = canvasRef.getContext("2d");
  const ctxo = overlayRef.getContext("2d");
  ctx.strokeStyle = "blue";
  ctx.lineWidth = 10;
  ctxo.strokeStyle = "blue";
  ctxo.lineWidth = 4;
  let canvasOffset = canvasRef.getBoundingClientRect();
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
    const mouseX = parseInt(e.clientX - offsetX);
    const mouseY = parseInt(e.clientY - offsetY);

    // Put your mousemove stuff here

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
    const roiObj = {
      label: selectedType,
      cords: {
        right_x: prevStartX,
        top_y: prevHeight + prevStartY,
        left_x: prevWidth + prevStartX,
        bottom_y: prevStartY,
      },
    };
  
    selectedRegions.push(roiObj);
    return selectedRegions;
  }



  function getSelectedRegions() 
  {
    return selectedRegions
  }





  return { handleMouseDown, handleMouseMove, handleMouseUp, handleMouseOut, getSelectedRegions };
}

const renderRectangle = renderRectangleFactory();

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



let selectedType = "anything";
const flask_url = "http://127.0.0.1:5000/api/cv/yolo";

//State:
const arrOfScreens = [videoInputRef, overlayRef, canvasRef, inputCanvasRef];

//Image input and returns a canvas version of the image (USED so I can draw on the image)
function drawCanvas(canvas, img) {
  canvas.width = getComputedStyle(canvas).width.split("px")[0];
  canvas.height = getComputedStyle(canvas).height.split("px")[0];
  let ratio = Math.min(canvas.width / img.width, canvas.height / img.height);
  let x = (canvas.width - img.width * ratio) / 2;
  let y = (canvas.height - img.height * ratio) / 2;
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  canvas
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
  output.src = data.img;
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

  //Updates the SRCs and Canvas in order to display Client Server Images
  renderVideo(data, imageCaptured);
}

const getVideo = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { min: 720 } },
  });

  //Gets the current screen
  const track = stream.getVideoTracks()[0];
  let { width, height } = track.getSettings();
  setSize(width, height);

  setInterval(() => {
    intervalProcessing(track);
  }, 1000);
};

getVideo();
