const demosSection = document.getElementById("demos");

var model = undefined;

// Before we can use COCO-SSD class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
cocoSsd.load().then(function (loadedModel) {
  model = loadedModel;
  console.log("READY");
});

const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Keep a reference of all the child elements we create
// so we can remove them easilly on each render.
var children = [];

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  const enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start classification.
function enableCam(event) {
  if (!model) {
    console.log("Wait! Model not loaded yet.");
    return;
  }

  // Hide the button.
  event.target.classList.add("removed");

  // getUsermedia parameters.
  const constraints = {
    video: { width: { min: 1280 }, height: { min: 720 } },
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// Prediction loop!
function predictWebcam() {
  // Now let's start classifying the stream.
  model.detect(video).then(function (predictions) {
    // Remove any highlighting we did previous frame.
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    children.splice(0);

    // Now lets loop through predictions and draw them to the live view if
    // they have a high confidence score.
    for (let n = 0; n < predictions.length; n++) {
      console.log(predictions);
      // If we are over 66% sure we are sure we classified it right, draw it!
      if (predictions[n].score > 0.66) {
        const p = document.createElement("p");
        p.innerText =
          predictions[n].class +
          " - with " +
          Math.round(parseFloat(predictions[n].score) * 100) +
          "% confidence.";
        // Draw in top left of bounding box outline.
        p.style =
          "left: " +
          predictions[n].bbox[0] +
          "px;" +
          "top: " +
          predictions[n].bbox[1] +
          "px;" +
          "width: " +
          (predictions[n].bbox[2] - 10) +
          "px;";

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
        highlighter.style =
          "left: " +
          predictions[n].bbox[0] +
          "px; top: " +
          predictions[n].bbox[1] +
          "px; width: " +
          predictions[n].bbox[2] +
          "px; height: " +
          predictions[n].bbox[3] +
          "px;";

        liveView.appendChild(highlighter);
        liveView.appendChild(p);

        // Store drawn objects in memory so we can delete them next time around.
        children.push(highlighter);
        children.push(p);
      }
    }

    // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
  });
}
