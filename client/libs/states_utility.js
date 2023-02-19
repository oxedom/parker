export function roiEvaluating(currentTime, firstCreated, differnce) {
  return currentTime - firstCreated < differnce ? true : false;
}
