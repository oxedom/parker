import { atom, selector } from "recoil";
import { checkRectOverlap, selectedFactory } from "../libs/utillity";
import {
  roiEvaluating,
  overlapsAndKnown,
  overlapsFirstDetect,
  calculateTimeDiff,
  supressedRoisProcess,
  convertRoisSelected,
} from "../libs/states_utility";
import { renderAllOverlaps, drawTextOnCanvas } from "../libs/canvas_utility";

const evaluateTimeState = atom({
  key: "evaluateTimeState",
  default: 5000,
});

const autoEvaluateTimeState = atom({
  key: "autoEvaluateTimeState",
  default: 10000,
});

const detectionThresholdState = atom({
  key: "detectionThresholdState",
  default: 0.6,
});

const overlapThresholdState = atom({
  key: "overlapThresholdState",
  default: 0.4,
});

const vehicleOnlyState = atom({
  key: "vehicleOnlyState",
  default: true,
});

const thresholdIouState = atom({
  key: "thresholdIouState",
  default: 0.65,
});

const showDetectionsState = atom({
  key: "showDetectionsState",
  default: true,
});

const fpsState = atom({
  key: "framesPerSecounds",
  default: 1,
});

const lastCheckedState = atom({
  key: "lastChecked",
  default: 0,
});

const processingState = atom({
  key: "processing",
  default: true,
});

const autoDetectState = atom({
  key: "autoDetect",
  default: false,
});

//Made for ROIS selected by hand or auto detected
const selectedRoi = atom({
  key: "selectedRois",
  default: [],
});

//Temo Arrays for FPS / Detection time.
const autoDetectArrState = atom({
  key: "autoDetectArrState",
  default: [],
});

const autoCheckedState = atom({
  key: "autoCheckedState",
  default: 0,
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
      let { cords } = action.payload;
      const roiObj = selectedFactory(cords);

      const oldRois = get(selectedRoi);

      const updatedArr = [...oldRois, roiObj];

      set(selectedRoi, updatedArr);
    }
    if (action.event === "deleteRoi") {
      let uid = action.payload;
      const oldRois = get(selectedRoi);
      const updatedArr = oldRois.filter((roi) => roi.uid !== uid);
      set(selectedRoi, updatedArr);
    }
    if (action.event === "occupation") {
      let { predictionsArr, canvas } = action.payload;

      let _lastChecked = get(lastCheckedState);
      let _autoDetect = get(autoDetectState);
      let _width = get(imageWidthState);
      let _height = get(imageHeightState);
      const currentUnixTime = Date.now();

      if (get(showDetectionsState) && predictionsArr.length > 0) {
        renderAllOverlaps(predictionsArr, canvas, _width, _height);
      }

      // This if statement prevents excessive calls to checkOverlap with ROIS.
      let excessiveCheck = Date.now() - _lastChecked > 900 && !_autoDetect;

      if (excessiveCheck) {
        set(lastCheckedState, Date.now());
      } else if (!_autoDetect) {
        return;
      }

      //If there are ROI to check objects the function returns
      const selectedRois = get(selectedRoi);
      if (selectedRois.length === 0 && !_autoDetect) {
        return;
      }

      //If no predections have happen, then a dummy predection is sent
      //so that the function runs and updates the selectedRois!
      if (predictionsArr.length === 0) {
        predictionsArr = [
          {
            cords: {
              right_x: -999,
              top_y: -999,
              width: -999,
              height: -999,
            },

            label: "car",
            confidenceLevel: 99,
            area: -999,
          },
        ];
      }

      const selectedRoisClone = structuredClone(selectedRois);

      const evaluateTime = get(evaluateTimeState);
      const overlapThreshold = get(overlapThresholdState);

      //   //Log N** function on quite a small scale so it's okay
      for (let index = 0; index < selectedRois.length; index++) {
        //Checking if the current
        let isOverlap = checkRectOverlap(
          selectedRois[index],
          predictionsArr,
          overlapThreshold
        );

        let roiNotEvaluating = !roiEvaluating(
          currentUnixTime,
          selectedRois[index]["time"],
          evaluateTime
        );

        if (roiNotEvaluating) {
          selectedRoisClone[index]["evaluating"] = false;
        }

        if (overlapsFirstDetect(isOverlap, selectedRois, index)) {
          //Update lastSeen and First Seen to current time
          selectedRoisClone[index]["firstSeen"] = currentUnixTime;
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;
        } else if (overlapsAndKnown(isOverlap, selectedRois, index)) {
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;
          //Calculate timeDIffernce
          let timeDiff = calculateTimeDiff(selectedRois, index);
          if (timeDiff > evaluateTime) {
            selectedRoisClone[index].occupied = true;
          }
        } else if (
          currentUnixTime - selectedRois[index]["lastSeen"] >
          evaluateTime
        ) {
          //Reset
          selectedRoisClone[index]["firstSeen"] = null;
          selectedRoisClone[index]["lastSeen"] = null;
          selectedRoisClone[index]["occupied"] = false;
        }
      }

      if (_autoDetect) {
        const autoChecked = get(autoCheckedState);
        const autoEvaluateTime = get(autoEvaluateTimeState)
        drawTextOnCanvas(canvas, _width, _height, "Auto Detcting");
        let adding = Date.now() - autoChecked;
        if (autoChecked === 0) {
          set(autoCheckedState, Date.now());
        } else if (adding <= autoEvaluateTime) {
          const autoDetectArr = get(autoDetectArrState);
          set(autoDetectArrState, [...autoDetectArr, predictionsArr]);
        } else {
          const autoDetectArr = get(autoDetectArrState);

          const suppresedRois = supressedRoisProcess(autoDetectArr, 0.8);

          const convertedToSelected = convertRoisSelected(suppresedRois);

          set(selectedRoi, convertedToSelected);
          set(autoDetectState, false);
          set(autoDetectArrState, []);
          set(autoCheckedState, 0);
        }
      } else {
        set(selectedRoi, selectedRoisClone);
      }

      //How long it takes to evaluate if a object is there or not
    }

    if (action.event === "selectRoi") {
      let uid = action.payload;
      //Array of ROI objects
      const selectedRois = get(selectedRoi);
      //Roi that needs to be toogled
      const targetRoi = selectedRois.filter((roi) => roi.uid === uid)[0];
      const targetRoiIndex = selectedRois.findIndex((roi) => roi.uid === uid);

      //Need to make copies
      const roiClone = structuredClone(targetRoi);
      const selectedRoisClone = structuredClone(selectedRois);

      roiClone.hover = true;
      selectedRoisClone[targetRoiIndex] = roiClone;
      set(selectedRoi, selectedRoisClone);
    }
    if (action.event === "unSelectRoi") {
      let uid = action.payload;
      //Array of ROI objects
      const selectedRois = get(selectedRoi);
      //Roi that needs to be toogled
      const targetRoi = selectedRois.filter((roi) => roi.uid === uid)[0];
      const targetRoiIndex = selectedRois.findIndex((roi) => roi.uid === uid);

      //Need to make copies
      const roiClone = structuredClone(targetRoi);
      const selectedRoisClone = structuredClone(selectedRois);

      roiClone.hover = false;

      selectedRoisClone[targetRoiIndex] = roiClone;
      set(selectedRoi, selectedRoisClone);
    }
    if (action.event === "deleteAllRois") {
      set(selectedRoi, []);
    }
  },
});

const imageHeightState = atom({
  key: "imageHeight",
  default: 480,
});

const imageWidthState = atom({
  key: "imageWidth",
  default: 640,
});

export {
  imageWidthState,
  imageHeightState,
  fpsState,
  processingState,
  evaluateTimeState,
  selectedRoiState,
  detectionThresholdState,
  thresholdIouState,
  showDetectionsState,
  vehicleOnlyState,
  autoDetectState,
};
