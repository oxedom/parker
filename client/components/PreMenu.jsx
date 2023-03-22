import { useEffect } from "react";
import { useRecoilState } from "recoil";
import { imageWidthState, imageHeightState } from "./states";
import { useRouter } from "next/navigation";

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
      case "webcam": {
        setTimeout(() => {
          setActivate(true);
        }, 200);
        break;
      }

      case "webrtc": {
        router.push("input");
      }
    }
  };

  return (
    <main className="text-white m-1 text-2xl sm:text-6xl gap-5 grid lg:grid-cols-2 grid-rows-2 md:grid-rows-none">

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
        Parker{" "}
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
        Stream{" "}
      </button>
    </main>
  );
};

export default PreMenu;
