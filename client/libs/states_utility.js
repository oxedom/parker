import { selectedFactory } from "./utillity";

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

export function supressedRoisProcess(roiMatrix) {
  console.log(roiMatrix);
  let candidates = getShortestArray(roiMatrix);
  console.log(candidates);

  return candidates;
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
