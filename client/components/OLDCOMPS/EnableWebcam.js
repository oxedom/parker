import { useEffect, useRef } from "react";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilState } from "recoil";
import { detectWebcam, getSetting } from "../libs/utillity";

const EnableWebcam = ({ setWebcamLoaded }) => {
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const enableWebcamRef = useRef(null);

  async function setUserSettings() {
    let { width, height } = await getSetting();

    setImageWidth(width);
    setImageHeight(height);
  }

  useEffect(() => {
    let loadingIntervalID;
    if (enableWebcamRef.current !== null) {
      let dotSring = "   ";
      let context = enableWebcamRef.current.getContext("2d");
      context.clearRect(0, 0, imageWidth, imageHeight);

      context.font = "bold 40px Arial";

      context.textAlign = "center";
      loadingIntervalID = setInterval(() => {
        if (dotSring === "   ") {
          dotSring = ".  ";
        } else if (dotSring === ".  ") {
          dotSring = ".. ";
        } else if (dotSring === ".. ") {
          dotSring = "...";
        } else if (dotSring === "...") {
          dotSring = "   ";
        }
        context.clearRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "blue";
        context.fill(0, 0, imageWidth, imageHeight);

        context.fillText(
          "Please Enable your webcam" + dotSring,
          imageWidth * 0.5,
          imageHeight * 0.3
        );
        context.font = "bold 28px Arial";
        context.fillText(
          "Make sure it's plugged in!   ",
          imageWidth * 0.5,
          imageHeight * 0.3 + 50
        );
        context.fillText(
          "Troubleshooting in the docs   ",
          imageWidth * 0.5,
          imageHeight * 0.3 + 100
        );
        context.font = "bold 40px Arial";
      }, 1000);
    }

    const intervalId = setInterval(() => {
      detectWebcam(async (hasWebcamBoolean) => {
        if (hasWebcamBoolean) {
          await setUserSettings().then(() => {
            setWebcamLoaded(true);
          });
        }
      });
    }, 1000);

    return () => {
      clearInterval(loadingIntervalID);
      clearInterval(intervalId);
    };
  }, []);

  return (
    <canvas
      width={imageWidth}
      ref={enableWebcamRef}
      height={imageHeight}
      className="flex justify-center items-center flex-1   bg-slate-100  "
    />
  );
};

export default EnableWebcam;
