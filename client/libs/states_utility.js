export function roiEvaluating(currentTime, firstCreated, differnce) {
  return currentTime - firstCreated < differnce ? true : false;
}

export function overlapsFirstDetect(isOverlap, selectedRois, index) {
  let firstSeen = selectedRois[index]['firstSeen']
  return (isOverlap && firstSeen === null)
}

export function overlapsAndKnown(isOverlap, selectedRois, index) {
  let firstSeen = selectedRois[index]['firstSeen']
  return (isOverlap && firstSeen !== null)
}

export  function calculateTimeDiff(selectedRois, index) 
{
  let timeDiff = selectedRois[index]["lastSeen"] - selectedRois[index]["firstSeen"];
  return timeDiff
}
