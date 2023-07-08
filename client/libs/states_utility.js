import { selectedFactory, getOverlap, filterArrayByScore } from "./utillity";

export function checkRoiEvaluating(currentTime, firstCreated, differnce) {
  return currentTime - firstCreated < differnce ? true : false;
}

export function firstDetect(isOverlap, selectedRois, index) {
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
//The supressedRoisProcess function takes in a matrix of rectangular regions of interest (ROIs) and a
//threshold value, and returns a filtered array of ROIs that meet the threshold criteria.
//It does this by first finding the longest array of ROIs in the matrix and creating an array of scores for each ROI
//in that array. It then loops through each ROI in the longest array and checks for overlaps with all
//other ROIs in the matrix. If there is a significant overlap (determined by a percentage difference threshold),
// the score for that ROI is incremented. Finally, the function filters the longest array by the score and threshold
// criteria, and returns the filtered array of ROIs.
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

export function  convertSuppressedRoisToSelected(arr) {
  return arr.map((a) => {
    return {
      ...selectedFactory(a.cords),
    };
  });
}

export function SnapshotFactory(selectedRois) {
  return selectedRois.map((s) => {
    const { evaluating, hover, label, cords, area, ...rest } = s;
    return rest;
  });
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
