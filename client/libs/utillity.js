import uniqid from "uniqid";
import labels from "./labels.json";
import { xywh2xyxy } from "./canvas_utility";

//The getOverlap function takes two rectangles as input and
// returns a new rectangle that represents the overlapping area between the two rectangles.
// If there is no overlap, the function returns null.
export function getOverlap(rectangle1, rectangle2) {
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

// The checkRectOverlap function takes three arguments: a rect object representing a rectangle to check for overlap with other rectangles,
// an array of detectionsArr containing other rectangles to compare with, and an overlapThreshold
// representing the minimum percentage of overlap required to consider two rectangles as overlapping. The function returns
// a boolean value indicating whether there is an overlap between the given rectangle and any of the rectangles in the detectionsArr array.
export function checkRectOverlap(rect, detectionsArr, overlapThreshold) {
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
      if (percentDiff > overlapThreshold) {
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
    label == "motorcycle" ||
    label === "bus"
  ) {
    return true;
  } else {
    return false;
  }
}

export function selectedFactory(cords) {
  let date = new Date();

  const roiObj = {
    label: "vehicle",
    cords: { ...cords },
    uid: uniqid() + "DATE" + Date.now().toString(),
    area: Math.round(cords.width * cords.height),
    firstSeen: null,
    lastSeen: null,
    occupied: null,
    events: [
      {
        eventName: "initialized",
        timeMarked: date.getTime(),
        duration: null,
      },
    ],
    cycleCount: 0,
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


export function filterArrayByScore(array, scores, threshold) {
  return array.filter((_, i) => scores[i] > threshold);
}

function shortenedCol(arrayofarray, indexlist) {
  return arrayofarray.map(function (array) {
    return indexlist.map(function (idx) {
      return array[idx];
    });
  });
}
