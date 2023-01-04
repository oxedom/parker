let input_imageCapture;
const canvasWraperRef = document.getElementById("canvasWrapper");
const videoInputRef = document.getElementById("input");
const overlayRef = document.getElementById("overlay");
const canvasRef = document.getElementById("canvas");
const inputCanvasRef = document.getElementById("input-canvas");
let sizeSet = false;
let cam_width = 640;
let selectedType = 'anything'
const flask_url = "http://127.0.0.1:5000/api/cv/yolo";

//State:
const selectedRegions = []


const arrOfScreens = [videoInputRef, overlayRef, canvasRef, inputCanvasRef];

const onTakePhotoButtonClick = async () => {
  try {
    const blob = await input_imageCapture.takePhoto();
    const imageBitmap = await createImageBitmap(blob);
    drawCanvas(inputCanvasRef, imageBitmap);
  } catch (error) {
    console.log(error);
  }
};

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

//Sets the size of all the canvas to be the same
const setSize = (width, height) => {
  arrOfScreens.forEach((screen) => {
    screen.width = width;
    screen.height = height;
  });
  sizeSet = true;
};

const getCapabilitiesOfDevice = async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  console.log(devices);
  const video_devices = devices.filter((d) => {
    return d.kind === "videoinput";
  });
  const videoDevice = video_devices[0];
  const videoDeviceCapabilities = await videoDevice.getCapabilities();
  console.log(videoDeviceCapabilities);
  return videoDeviceCapabilities;
};



const getVideo = () => {
  navigator.mediaDevices
    .getUserMedia({ video: { width: { min: 1280 } } })
    .then(async (stream) => {
      //Gets the current screen
      let video = document.getElementById("input");
      //Sets the Src of video Object
      video.srcObject = stream;
      //Plays the video
      video.play();

      //Interval that captures image from Steam, converts to array buffer and
      //sends it to API as a bufferArray, waiting for response and sets the base64 to
      //the image SRC on output
      setInterval(async () => {
        const track = stream.getVideoTracks()[0];

        let { width, height } = track.getSettings();

        if (!sizeSet) {
          setSize(width, height);
        }
        let imageCapture = new ImageCapture(track);
        input_imageCapture = imageCapture;
        const capturedImage = await imageCapture.takePhoto();

        let imageBuffer = await capturedImage.arrayBuffer();
        imageBuffer = new Uint8Array(imageBuffer);
        //Sets the Canvas to the current Image that has been capatured
        onTakePhotoButtonClick();

        const res = await fetch(flask_url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ buffer: [...imageBuffer] }),
        });

        const output = document.getElementById("output");

        const resJson = await res.json();
          resJson.meta_data.detections.forEach(detection =>
    {   
        const obj = 
        {
            top_x: detection.top_left_cords.top_x,
            top_y: detection.top_left_cords.top_y,
            bottom_x: detection.bottom_right_cords.bottom_x,
            bottom_y:detection.bottom_right_cords.bottom_y
        }
        if(selectedRegions.length > 0) 
        {
            selectedRegions.forEach(selected => 
                {
                    
                    const a = checkIfWithin(obj, selected.cords)
               
                })
        }

        
        // ctx.fill();
    })
        output.src = resJson.img;
      }, 1000);
    })

    .catch((err) => {
      console.error("error:", err);
    });
};

getVideo();

// get references to the canvas and context
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

function addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight) 
{
const roiObj = 
{
    type:  selectedType,
    cords: {
        top_x: prevStartX, 
        top_y:  prevHeight+prevStartY, 
        bottom_x: prevWidth+prevStartX,
        bottom_y: prevStartY
    }
}    

selectedRegions.push(roiObj)
console.log(selectedRegions[0].cords)
}

function handleMouseUp(e) {
  e.preventDefault();
  e.stopPropagation();

  // the drag is over, clear the dragging flag
  isDown = false;


  ctxo.strokeRect(prevStartX, prevStartY, prevWidth, prevHeight);
  addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight)  
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

canvasRef.addEventListener("mousedown", (e) => {
  e.preventDefault();
  handleMouseDown(e);
});

canvasRef.addEventListener("mousemove", (e) => {
  e.preventDefault();
  handleMouseMove(e);
});

canvasRef.addEventListener("mouseout", (e) => {
  e.preventDefault();
  handleMouseOut(e);
});

canvasRef.addEventListener("mouseup", (e) => {
  e.preventDefault();
  handleMouseUp(e);
});

function checkIfWithin(mother, child) {
    // Calculate the area of the mother rectangle

   

    // Calculate the area of the child rectangle

  
    // // Calculate the overlapping area between the two rectangles

  
    // Calculate the percentage of the child rectangle that is contained within the mother rectangle

  
    return 100;
  }