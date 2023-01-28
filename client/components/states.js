import { atom, selector } from "recoil";
import uniqid from "uniqid";
import { finalName } from "../libs/utillity";

const selectedRoi = atom({
  key: "selectedRois",
  default: [],
});

const roiTypeState = atom({
  key: "roiType",
  default: "Any",
});

const roiNameState = atom({
  key: "roiName",
  default: "",
});

const processingState = atom({
  key: "processing",
  default: false,
});

const selectedRoiState = selector({
  key: "selectedRoisState",
  default: [],
  get: ({ get }) => {
    const selectedRois = get(selectedRoi);
    return selectedRois;
  },

  set: ({ set, get }, action) => {
    if (action.event === "addRoi") {
      let { cords, color } = action.payload;
      let date = new Date();
      let roiType = get(roiTypeState);
      let roiName = get(roiNameState);
      const oldRois = get(selectedRoi);

      const roiObj = {
        name: finalName(roiName, oldRois.length),
        label: roiType,
        color: color,
        occupied: false,
        cords: { ...cords },
        time: date.getTime(),
        uid: uniqid(),
      };

      const updatedArr = [...oldRois, roiObj];
      console.log(roiObj);
      set(selectedRoi, updatedArr);
    }
    if (action.event === "deleteRoi") {
      let uid = action.payload;
      const oldRois = get(selectedRoi);
      const updatedArr = oldRois.filter((roi) => roi.uid !== uid);
      set(selectedRoi, updatedArr);
    }

    if (action.event === "toogleRoiByID") {
      let uid = action.payload;
      //Array of ROI objects
      const currentRois = get(selectedRoi);
      //Roi that needs to be toogled
      const targetRoi = currentRois.filter((roi) => roi.uid === uid);
      const targetRoiIndex = currentRois.findIndex((roi) => roi.uid === uid);

      //Toogle occupied
      targetRoi.occupied = !targetRoi.occupied;
      currentRois[targetRoiIndex] = targetRoi;

      set(selectedRoi, currentRois);
    }
  },
});
// const track = useRecoilValue(track);
const trackState = atom({
  key: "track",
  default: null,
});

const selectingColorState = atom({
  key: "selectingColor",
  default: "#FF0000",
});

const selectedColorState = atom({
  key: "selectedColor",
  default: "#FF0000",
});

const detectionColorState = atom({
  key: "detectionColor",
  default: "#0000FF",
});

const imageHeightState = atom({
  key: "imageHeight",
  default: 720,
});

const imageWidthState = atom({
  key: "imageWidth",
  default: 1280,
});

const outputImageState = atom({
  key: "outputImage",
  default: "",
});

export {
  roiTypeState,
  trackState,
  roiNameState,
  imageWidthState,
  imageHeightState,
  detectionColorState,
  selectingColorState,
  selectedColorState,
  processingState,
  outputImageState,
  selectedRoiState,
};
