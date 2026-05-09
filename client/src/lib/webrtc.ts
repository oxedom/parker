function createEmptyAudioTrack(): MediaStreamTrack {
  const ctx = new AudioContext();
  const oscillator = ctx.createOscillator();
  const dst = oscillator.connect(ctx.createMediaStreamDestination());
  oscillator.start();
  const track = (dst as MediaStreamAudioDestinationNode).stream.getAudioTracks()[0];
  track.enabled = false;
  return track;
}

function createEmptyVideoTrack(width: number, height: number): MediaStreamTrack {
  const canvas = Object.assign(document.createElement("canvas"), { width, height });
  canvas.getContext("2d")!.fillRect(0, 0, width, height);
  const stream = canvas.captureStream();
  const track = stream.getVideoTracks()[0];
  track.enabled = false;
  return track;
}

export function createEmptyStream(width = 640, height = 480): MediaStream {
  return new MediaStream([createEmptyAudioTrack(), createEmptyVideoTrack(width, height)]);
}

export function detectWebcam(): Promise<boolean> {
  const md = navigator.mediaDevices;
  if (!md?.enumerateDevices) return Promise.resolve(false);
  return md.enumerateDevices().then((devices) =>
    devices.some((d) => d.kind === "videoinput"),
  );
}

export async function getDefaultVideoSettings(): Promise<{
  width: number;
  height: number;
}> {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  const settings = stream.getTracks()[0].getSettings();
  stream.getTracks().forEach((t) => t.stop());
  return {
    width: settings.width ?? 640,
    height: settings.height ?? 480,
  };
}
