const flask_url = "http://127.0.0.1:5000/api/cv/yolo";
import * as tf from '@tensorflow/tfjs';
const cocoSsd = require('@tensorflow-models/coco-ssd');

function getOverlap(rectangle1, rectangle2) {
  const intersectionX1 = Math.max(rectangle1.right_x, rectangle2.right_x);
  const intersectionX2 = Math.min(
    rectangle1.right_x + rectangle1.width,
    rectangle2.right_x + rectangle2.width
  );
  if (intersectionX2 < intersectionX1) {
    return null;
  }
  const intersectionY1 = Math.max(rectangle1.top_y, rectangle2.top_y);
  const intersectionY2 = Math.min(
    rectangle1.top_y + rectangle1.height,
    rectangle2.top_y + rectangle2.height
  );
  if (intersectionY2 < intersectionY1) {
    return null;
  }

  return {
    right_x: intersectionX1,
    top_y: intersectionY1,
    width: intersectionX2 - intersectionX1,
    height: intersectionY2 - intersectionY1,
    area: (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1),
  };
}

async function capturedImageToBuffer(capturedImage) {
  const imagePhoto = await capturedImage.takePhoto();

  let imageBuffer = await imagePhoto.arrayBuffer();

  imageBuffer = new Uint8Array(imageBuffer);

  return imageBuffer;
}

export async function capturedImageServer(capturedImage) {
  const imageBuffer = await capturedImageToBuffer(capturedImage);

  const res = await fetch(flask_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ buffer: [...imageBuffer] }),
  });

  const data = await res.json();

  return data;
}

function rectangleArea(rect) {
  const width = rect.cords.right_x - rect.cords.left_x;
  const height = rect.cords.bottom_y - rect.cords.top_y;
  return Math.abs(width * height);
}

export function finalName(name, arrayLength) {
  if (name === "") {
    return `ROI NUMBER: ${arrayLength + 1}`;
  } else return name;
}

export function checkOverlapArrays(detectionsArr, selectedArr) 
{
let overlaps = []
  detectionsArr.forEach(d => 
    {
      selectedArr.forEach(s => 
        {
        let overlapCords = getOverlap(d.cords, s.cords)
        let selectedArea = s.cords.width * s.cords.height
        console.log("Selected area is ", selectedArea);
        console.log("OverlapCords area is", overlapCords.area);
        
        if(overlapCords != null) {  
          let overlap = 
          {
            ...s,
            color: "#FFEF00",
            cords: overlapCords
          }
          console.log(overlap);
          overlaps.push(overlap)
        }
       
          

        })
    })
return overlaps
}


export  async function loadModel() {
  const model = await cocoSsd.load()
  console.log("Model loaded");
  return model;
}

export function predictWebcam(model, video) {
  // Now let's start classifying the stream.

  model.detect(video).then(function (predictions) {
    // Remove any highlighting we did previous frame.
    // for (let i = 0; i < children.length; i++) {
    //   liveView.removeChild(children[i]);
    // }
    // children.splice(0);

    // Now lets loop through predictions and draw them to the live view if
    // they have a high confidence score.
    for (let n = 0; n < predictions.length; n++) {
      console.log(predictions);
      // If we are over 66% sure we are sure we classified it right, draw it!
      if (predictions[n].score > 0.66) {
        // const p = document.createElement("p");
        // p.innerText =
        //   predictions[n].class +
        //   " - with " +
        //   Math.round(parseFloat(predictions[n].score) * 100) +
        //   "% confidence.";
        // Draw in top left of bounding box outline.
        // p.style =
        //   "left: " +
        //   predictions[n].bbox[0] +
        //   "px;" +
        //   "top: " +
        //   predictions[n].bbox[1] +
        //   "px;" +
        //   "width: " +
        //   (predictions[n].bbox[2] - 10) +
        //   "px;";

        // Draw the actual bounding box.
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        const left_x = predictions[n].bbox[0];
        const top_y = predictions[n].bbox[1];
        const width = predictions[n].bbox[2];
        const height = predictions[n].bbox[3];
        console.log(predictions);
        const obj = { left_x, top_y, width, height };
        console.log(obj);
        const svg = document.createElement("svg");
        const rect = document.createElement("rect");
        // rect.setAttribute('x')
        // rect.setAttribute('y')
        // rect.setAttribute('width')
        // rect.setAttribute('height')
        // rect.setAttribute('x')
        // rect.setAttribute('x')
        // rect.setAttribute('fill')
        // highlighter.style =
        //   "left: " +
        //   predictions[n].bbox[0] +
        //   "px; top: " +
        //   predictions[n].bbox[1] +
        //   "px; width: " +
        //   predictions[n].bbox[2] +
        //   "px; height: " +
        //   predictions[n].bbox[3] +
        //   "px;";

        // liveView.appendChild(highlighter);
        // liveView.appendChild(p);

        // // Store drawn objects in memory so we can delete them next time around.
        // children.push(highlighter);
        // children.push(p);
      }
    }

    // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
  });
}
