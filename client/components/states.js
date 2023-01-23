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
      console.log(action.payload);
      let { cords, color } = action.payload;
      let date = new Date();
      let roiType = get(roiTypeState);
      let roiName = get(roiNameState);
      const oldRois = get(selectedRoi);

      const roiObj = {
        name: finalName(roiName, oldRois.length),
        label: roiType,
        color: color,
        cords: { ...cords },
        time: date.getTime(),
        uid: uniqid(),
      };

      const updatedArr = [...oldRois, roiObj];

      set(selectedRoi, updatedArr);
    }
    if (action.event === "deleteRoi") {
      let uid = action.payload;
      const oldRois = get(selectedRoi);
      const updatedArr = oldRois.filter((roi) => roi.uid !== uid);
      set(selectedRoi, updatedArr);
    }
  },
});

const selectingColorState = atom({
  key: "selectingColor",
  default: "#EC3945",
});

const selectedColorState = atom({
  key: "selectedColor",
  default: "#8FC93A",
});

const detectionColorState = atom({
  key: "detectionColor",
  default: "#FF0000",
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
