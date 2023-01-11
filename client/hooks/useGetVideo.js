import { useState, useEffect } from "react";

const useGetVideo = (width = 1280, height = 720) => {
  const [stream, setStream] = useState(null);
  const [track, setTrack] = useState(null);

  useEffect(() => {
    const getVideo = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { min: image_width } },
      });

      setStream(stream);
      setTrack(stream.getVideoTracks()[0]);
      const { width, height } = track.getSettings();
      setWidth(width);
      setHeight(height);
    };

    getVideo();
  }, []);

  return { stream, track, width, height };
};
