import { useEffect } from "react";

const Webcam = ({ canvasRef, overlayRef, outputRef, inputCanvasRef }) => {
  const flask_url = "http://127.0.0.1:5000/api/cv/yolo";
const arrOfRefs = [canvasRef,overlayRef,outputRef,inputCanvasRef]

function setSize (width, height) {
  arrOfRefs.forEach((screen) => {
    if(screen !== null) 
    {
      screen.width = width;
      screen.height = height;
    }
 
  });
};

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
  console.log(data);
 
  return data;
}

function renderVideo(data, imageCaptured) {
  onTakePhotoButtonClick(imageCaptured);
  if(outputRef !== null) 
  {
    outputRef.src = data.img;
  }

}

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



async function intervalProcessing(track) 
{
  //Converts imageCaptured parameter to buffer and sends it to the server for computer vision
  //processing and returns an object with a new image with meta_data after processing
  const imageCaptured = new ImageCapture(track);
  const data = await bufferToServer(imageCaptured);
  renderVideo(data, imageCaptured);
}


async function getVideo()
{
  const stream = await navigator.mediaDevices.getUserMedia({video: { width: { min: 640 } },});
  const track = stream.getVideoTracks()[0];

  let { width, height } = track.getSettings();
  setSize(width, height);

  setTimeout(() => {
    intervalProcessing(track);
  }, 10);

}

getVideo()




  return <div>
    <h1> I am video</h1>
  </div>;
};


export async function getStaticProps() {
  // Call an external API endpoint to get posts.
  // You can use any data fetching library


  // By returning { props: { posts } }, the Blog component
  // will receive `posts` as a prop at build time
  return {
    props: {
      canvasRef,
      overlayRef,
      outputRef,
      inputCanvasRef
  }
}

}


export default Webcam;
