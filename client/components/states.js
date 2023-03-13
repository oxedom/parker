import { atom, selector } from "recoil";
import { checkRectOverlap, selectedFactory } from "../libs/utillity";
import {
  roiEvaluating,
  overlapsAndKnown,
  overlapsFirstDetect,
  calculateTimeDiff,
} from "../libs/states_utility";
import { renderAllOverlaps } from "../libs/canvas_utility";

const evaluateTimeState = atom({
  key: "evaluateTimeState",
  default: 5000,
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

const selectedRoi = atom({
  key: "selectedRois",
  default: [],
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

      if (_autoDetect) {
        let updatedArr = [];
        predictionsArr.forEach((pred) => {
          let roiObj = selectedFactory(pred.cords);

          updatedArr.push(roiObj);
        });
        set(selectedRoi, updatedArr);

        set(autoDetectState, false);
      }

      if (get(showDetectionsState) && predictionsArr.length > 0) {
        renderAllOverlaps(predictionsArr, canvas, _width, _height);
      }

      // This if statement prevents excessive calls to checkOverlap with ROIS.
      let excessiveCheck = Date.now() - _lastChecked > 900 && !_autoDetect;

      if (excessiveCheck) {
        set(lastCheckedState, Date.now());
      } else {
        return;
      }

      //If there are ROI to check objects the function returns
      const selectedRois = get(selectedRoi);
      if (selectedRois.length === 0) {
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
      const overlapThreshold = get(overlapThresholdState)

      //   //Log N** function on quite a small scale so it's okay
      for (let index = 0; index < selectedRois.length; index++) {
        //Checking if the current
        let isOverlap = checkRectOverlap(selectedRois[index], predictionsArr, overlapThreshold);

        let roiNotEvaluating = !roiEvaluating(
          currentUnixTime,
          selectedRois[index]["time"],
          evaluateTime
        );

        if(roiNotEvaluating) {selectedRoisClone[index]["evaluating"] = false}

        if (overlapsFirstDetect(isOverlap, selectedRois, index)) {
          //Update lastSeen and First Seen to current time
          selectedRoisClone[index]["firstSeen"] = currentUnixTime;
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;
        } else if (overlapsAndKnown(isOverlap, selectedRois, index)) {
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;
          //Calculate timeDIffernce
          let timeDiff = calculateTimeDiff(selectedRois, index)
          if(timeDiff > evaluateTime) { (selectedRoisClone[index].occupied = true)}
   
         
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

      set(selectedRoi, selectedRoisClone);
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
