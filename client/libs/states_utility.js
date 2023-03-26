import { selectedFactory, getOverlap, filterArrayByScore } from "./utillity";

export function roiEvaluating(currentTime, firstCreated, differnce) {
  return currentTime - firstCreated < differnce ? true : false;
}

export function overlapsFirstDetect(isOverlap, selectedRois, index) {
  let firstSeen = selectedRois[index]["firstSeen"];
  return isOverlap && firstSeen === null;
}

export function overlapsAndKnown(isOverlap, selectedRois, index) {
  let firstSeen = selectedRois[index]["firstSeen"];
  return isOverlap && firstSeen !== null;
}

export function calculateTimeDiff(selectedRois, index) {
  let timeDiff =
    selectedRois[index]["lastSeen"] - selectedRois[index]["firstSeen"];
  return timeDiff;
}

export function supressedRoisProcess(roiMatrix, threshold) {
  let longestArrPos = getLongestArray(roiMatrix);
  let longestArr = roiMatrix[longestArrPos];

  let scores = [];
  for (let index = 0; index < longestArr.length; index++) {
    scores[index] = 0;
  }

  for (let i = 0; i < longestArr.length; i++) {
    const currentPotential = longestArr[i];

    roiMatrix.forEach((dect) => {
      dect.forEach((d) => {
        let overlapCords = getOverlap(currentPotential.cords, d.cords);
        if (overlapCords != null) {
          let overlapArea_rounded = Math.round(overlapCords.area);
          let currentPotential_rounded = Math.round(currentPotential.area);

          let percentDiff = overlapArea_rounded / currentPotential_rounded;

          if (
            percentDiff > 0.8 ||
            overlapArea_rounded === currentPotential_rounded
          ) {
            scores[i] = scores[i] + 1;
          }
        }
      });
    });
  }

  let filterThreshold = Math.ceil(roiMatrix.length * threshold);
  let filtered = filterArrayByScore(longestArr, scores, filterThreshold);

  return filtered;
}

export function convertRoisSelected(arr) {
  let updatedArr = [];
  arr.forEach((pred) => {
    let roiObj = selectedFactory(pred.cords);
    updatedArr.push(roiObj);
  });

  return updatedArr;
}

function getShortestArray(arr) {
  let shortest = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i].length < shortest.length) {
      shortest = arr[i];
    }
  }
  return shortest;
}

function getLongestArray(arr) {
  let maxLength = 0;
  let longestArrayPos = null;

  for (let i = 0; i < arr.length; i++) {
    if (arr[i].length > maxLength) {
      maxLength = arr[i].length;
      longestArrayPos = i;
    }
  }
  return longestArrayPos;
}
