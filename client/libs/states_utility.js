export function roiEvaluating(currentTime, firstCreated, differnce) {
  console.log(currentTime - firstCreated);
  return currentTime - firstCreated < differnce ? true : false;
}
