import { exp } from "@tensorflow/tfjs";
import uniqid from "uniqid";
import labels from "../utils/labels.json";
import { xywh2xyxy } from "../utils/renderBox.js";

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

export function checkOverlapArrays(detectionsArr, selectedArr) {
  let overlaps = [];
  detectionsArr.forEach((d) => {
    selectedArr.forEach((s) => {
      let overlapCords = getOverlap(d.cords, s.cords);
      if (overlapCords != null) {
        let overlap = {
          ...s,
          cords: overlapCords,
        };

        overlaps.push(overlap);
      }
    });
  });
  return overlaps;
}

export function checkRectOverlap(rect, detectionsArr) {
  let answer = false;
  detectionsArr.forEach((d) => {
    //If answer is already true return answer
    if (answer == true) {
      return answer;
    }
    ///Overlap calculation
    let overlapCords = getOverlap(d.cords, rect.cords);
    //If overlapcords is null the squares don't intersect
    if (overlapCords == null) {
      return;
    } else {
      let overlapArea_rounded = Math.round(overlapCords.area);
      let rectArea_rounded = Math.round(rect.area);

      //If the overlap is the same size (The decection is bigger than the selection)
      if (overlapArea_rounded == rectArea_rounded) {
        answer = true;
      }

      //Overlap rounded should be smaller than rectArea, so we calcualte
      //how much % of the sqaure it's overlapping
      let percentDiff = overlapArea_rounded / rectArea_rounded;

      //If it overlaps more than 40% of the square return true, else false
      if (percentDiff > 0.4) {
        answer = true;
      } else {
        return;
      }
    }
  });
  return answer;
}

export function isVehicle(label) {
  if (
    label === "car" ||
    label === "truck" ||
    "label" == "motorcycle" ||
    label === "bus"
  ) {
    return true;
  } else {
    return false;
  }
}

export const getSetting = async () => {
  let stream = await navigator.mediaDevices.getUserMedia({ video: true });
  let { width, height } = stream.getTracks()[0].getSettings();
  return { width, height };
};

export function detectWebcam(callback) {
  let md = navigator.mediaDevices;
  if (!md || !md.enumerateDevices) return callback(false);
  md.enumerateDevices().then((devices) => {
    callback(devices.some((device) => "videoinput" === device.kind));
  });
}

export function selectedFactory(cords) {
  let date = new Date();

  const roiObj = {
    label: "vehicle",
    cords: { ...cords },
    time: date.getTime(),
    uid: uniqid(),
    area: cords.width * cords.height,
    firstSeen: null,
    lastSeen: null,
    occupied: null,
    hover: false,
    evaluating: true,
  };

  return roiObj;
}

export function totalOccupied(roiArr) {
  let OccupiedCount = 0;
  let availableCount = 0;
  roiArr.forEach((roi) => {
    if (roi.occupied === true) {
      OccupiedCount++;
    }
    if (roi.occupied === false) {
      availableCount++;
    }
  });
  return { OccupiedCount, availableCount };
}

export function webcamRunning() {
  if (
    typeof webcamRef.current !== "undefined" &&
    webcamRef.current !== null &&
    webcamRef.current.video.readyState === 4
  ) {
    return true;
  } else {
    return false;
  }
}

export function detectionsToROIArr(
  detectionIndices,
  boxes,
  class_detect,
  scores,
  imageWidth,
  imageHeight,
  vehicleOnly
) {
  let _predictionsArr = [];
  if (detectionIndices.length < 0) {
    return [];
  }

  for (let i = 0; i < detectionIndices.length; i++) {
    const detectionIndex = detectionIndices[i];
    const detectionScore = scores[detectionIndex];
    const detectionClass = class_detect[detectionIndex];
    let dect_label = labels[detectionClass];

    let condition = true 
    vehicleOnly ? condition = isVehicle(dect_label) : null

    
    if (condition) {
      const roiObj = { cords: {} };
      let [x1, y1, x2, y2] = xywh2xyxy(boxes[detectionIndex]);

      // Extract the bounding box coordinates from the 'boxes' tensor
      y1 = y1 * (imageHeight / 640);
      y2 = y2 * (imageHeight / 640);
      x1 = x1 * (imageWidth / 640);
      x2 = x2 * (imageWidth / 640);

      let dect_width = x2 - x1;
      let dect_weight = y2 - y1;
      roiObj.cords.bottom_y = y2;
      roiObj.cords.left_x = x2;
      roiObj.cords.top_y = y1;
      roiObj.cords.right_x = x1;
      roiObj.cords.width = dect_width;
      roiObj.cords.height = dect_weight;
      // Add the detection score to the bbox object
      roiObj.confidenceLevel = detectionScore;
      roiObj.label = dect_label;
      roiObj.area = dect_width * dect_weight;
      // Add the bbox object to the bboxes array

      _predictionsArr.push(roiObj);
    }
  }

  return _predictionsArr;
}
