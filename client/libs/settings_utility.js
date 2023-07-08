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
