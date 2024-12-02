// import uniqid from 'uniqid'
const labels = [
  'human',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush'
]
import { xywh2xyxy } from './canvas'
import type { RoiObject, Rectangle } from '../types'

export function detectionsToROIArr(
  detections: Array<RoiObject>,
  imageWidth: number,
  imageHeight: number,
  vehicleOnly: boolean
) {
  let condition = false;

  let _predictionsArr = [];
  const boxes = shortenedCol(detections, [0, 1, 2, 3]);
  const scores = shortenedCol(detections, [4]);
  const class_detect = shortenedCol(detections, [5]);
  if (detections === undefined || detections.length <= 0) {
    return [];
  }

  for (let index = 0; index < detections.length; index++) {
    const detectionScore = scores[index];
    const detectionClass = class_detect[index];
    let dect_label = labels[detectionClass];
    if (vehicleOnly === false) {
      condition = true;
    } else {
      condition = isVehicle(dect_label);
    }
    if (condition) {
      const roiObj: RoiObject = {
        cords: {
          bottom_y: null,
          left_x: null,
          top_y: null,
          right_x: null,
          width: null,
          height: null,
        },
        confidenceLevel: null,
        label: null,
        area: null,
      };
      let [x1, y1, x2, y2] = xywh2xyxy(boxes[index]);
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


//The getOverlap function takes two rectangles as input and
// returns a new rectangle that represents the overlapping area between the two rectangles.
// If there is no overlap, the function returns null.
export function getOverlap(rectangle1: Rectangle, rectangle2: Rectangle) {



  const intersectionX1 = Math.max(rectangle1.right_x, rectangle2.right_x)
  const intersectionX2 = Math.min(
    rectangle1.right_x + rectangle1.width,
    rectangle2.right_x + rectangle2.width
  )
  if (intersectionX2 < intersectionX1) {
    return null
  }
  const intersectionY1 = Math.max(rectangle1.top_y, rectangle2.top_y)
  const intersectionY2 = Math.min(
    rectangle1.top_y + rectangle1.height,
    rectangle2.top_y + rectangle2.height
  )
  if (intersectionY2 < intersectionY1) {
    return null
  }

  return {
    right_x: intersectionX1,
    top_y: intersectionY1,
    width: intersectionX2 - intersectionX1,
    height: intersectionY2 - intersectionY1,
    area: (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1)
  }
}

export function checkOverlapArrays(detectionsArr, selectedArr) {
  let overlaps = []
  detectionsArr.forEach((d) => {
    selectedArr.forEach((s) => {
      let overlapCords = getOverlap(d.cords, s.cords)
      if (overlapCords != null) {
        let overlap = {
          ...s,
          cords: overlapCords
        }

        overlaps.push(overlap)
      }
    })
  })
  return overlaps
}

// The checkRectOverlap function takes three arguments: a rect object representing a rectangle to check for overlap with other rectangles,
// an array of detectionsArr containing other rectangles to compare with, and an overlapThreshold
// representing the minimum percentage of overlap required to consider two rectangles as overlapping. The function returns
// a boolean value indicating whether there is an overlap between the given rectangle and any of the rectangles in the detectionsArr array.
export function checkRectOverlap(rect, detectionsArr, overlapThreshold) {
  let answer = false
  detectionsArr.forEach((d) => {
    //If answer is already true return answer
    if (answer == true) {
      return answer
    }
    ///Overlap calculation
    let overlapCords = getOverlap(d.cords, rect.cords)
    //If overlapcords is null the squares don't intersect
    if (overlapCords == null) {
      return
    } else {
      let overlapArea_rounded = Math.round(overlapCords.area)
      let rectArea_rounded = Math.round(rect.area)

      //If the overlap is the same size (The decection is bigger than the selection)
      if (overlapArea_rounded == rectArea_rounded) {
        answer = true
      }

      //Overlap rounded should be smaller than rectArea, so we calcualte
      //how much % of the sqaure it's overlapping
      let percentDiff = overlapArea_rounded / rectArea_rounded

      //If it overlaps more than 40% of the square return true, else false
      if (percentDiff > overlapThreshold) {
        answer = true
      } else {
        return
      }
    }
  })
  return answer
}

export function isVehicle(label) {
  if (label === 'car' || label === 'truck' || label == 'motorcycle' || label === 'bus') {
    return true
  } else {
    return false
  }
}

export function selectedFactory(cords) {
  let date = new Date()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx'.replace(/[x]/g, () =>
    ((Math.random() * 16) | 0).toString(16)
  )

  const roiObj = {
    uuid,
    label: 'vehicle',
    cords: { ...cords },
    area: Math.round(cords.width * cords.height),
    firstSeen: null,
    lastSeen: null,
    occupied: null,
    events: [
      {
        eventName: 'initialized',
        timeMarked: date.getTime(),
        duration: null
      }
    ],
    cycleCount: 0,
    hover: false,
    evaluating: true
  }

  return roiObj
}

export function totalOccupied(roiArr) {
  let OccupiedCount = 0
  let availableCount = 0
  roiArr.forEach((roi) => {
    if (roi.occupied === true) {
      OccupiedCount++
    }
    if (roi.occupied === false) {
      availableCount++
    }
  })
  return { OccupiedCount, availableCount }
}

export function detectionsToRoi(detections, imageWidth, imageHeight, vehicleOnly) {
  let condition = false

  let _predictionsArr = []
  const boxes = shortenedCol(detections, [0, 1, 2, 3])
  const scores = shortenedCol(detections, [4])
  const class_detect = shortenedCol(detections, [5])
  if (detections === undefined || detections.length <= 0) {
    return []
  }

  for (let index = 0; index < detections.length; index++) {
    const detectionScore = scores[index]
    const detectionClass = class_detect[index]
    let dect_label = labels[detectionClass]
    if (vehicleOnly === false) {
      condition = true
    } else {
      condition = isVehicle(dect_label)
    }
    if (condition) {
      const roiObj = { cords: {} }
      let [x1, y1, x2, y2] = xywh2xyxy(boxes[index])
      // Extract the bounding box coordinates from the 'boxes' tensor
      y1 = y1 * (imageHeight / 640)
      y2 = y2 * (imageHeight / 640)
      x1 = x1 * (imageWidth / 640)
      x2 = x2 * (imageWidth / 640)
      let dect_width = x2 - x1
      let dect_weight = y2 - y1
      roiObj.cords.bottom_y = y2
      roiObj.cords.left_x = x2
      roiObj.cords.top_y = y1
      roiObj.cords.right_x = x1
      roiObj.cords.width = dect_width
      roiObj.cords.height = dect_weight
      // Add the detection score to the bbox object
      roiObj.confidenceLevel = detectionScore
      roiObj.label = dect_label
      roiObj.area = dect_width * dect_weight
      // Add the bbox object to the bboxes array

      _predictionsArr.push(roiObj)
    }
  }

  return _predictionsArr
}

export function filterArrayByScore(array, scores, threshold) {
  return array.filter((_, i) => scores[i] > threshold)
}

function shortenedCol(arrayofarray, indexlist) {
  return arrayofarray.map(function (array) {
    return indexlist.map(function (idx) {
      return array[idx]
    })
  })
}
