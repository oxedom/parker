import { useEffect } from "react";
import { useRecoilState } from "recoil";
import { imageWidthState, imageHeightState } from "./states";
import { useRouter } from 'next/navigation';

const PreMenu = ({ setActivate, setDemo }) => {
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const router = useRouter();
  useEffect(() => {
    setImageWidth(640);
    setImageHeight(480);
  }, []);

  const handleClick = (value) => {
    switch (value) {
      case "demo": {
        setTimeout(() => {
          setDemo(true);
        }, 200);
        break;
      }
      case "webcam": {
        setTimeout(() => {
          setActivate(true);
        }, 200);
        break;
      }

      case "webrtc": 
      {
        router.push("output")
      }
    }
  };

  return (
    <main className="text-white m-1 text-2xl sm:text-6xl gap-5 grid lg:grid-cols-3 grid-rows-3 md:grid-rows-none">
      <button
        value={"demo"}
        onClick={(e) => {
          handleClick(e.target.value);
        }}
        onTouchStart={(e) => {
          handleClick(e.target.value);
        }}
        className="bg-orange-600 p-5 rounded-lg hover:scale-105 duration-300"
      >
        Demo{" "}
      </button>
      <button
        value={"webcam"}
        onClick={(e) => {
          handleClick(e.target.value);
        }}
        onTouchStart={(e) => {
          handleClick(e.target.value);
        }}
        className="bg-orange-600 p-5 rounded-lg hover:scale-105 duration-300"
      >
        Use Webcam{" "}
      </button>

      <button
        value={"webrtc"}
        onClick={(e) => {
          handleClick(e.target.value);
        }}
        onTouchStart={(e) => {
          handleClick(e.target.value);
        }}
        className="bg-orange-600 p-5 rounded-lg hover:scale-105 duration-300"
      >
        Livestream{" "}
      </button>
    </main>
  );
};

export default PreMenu;
