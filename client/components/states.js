import { atom, selector } from "recoil";
import { checkRectOverlap, selectedFactory } from "../libs/utillity";
import { roiEvaluating } from "../libs/states_utility";
import {  renderAllOverlaps } from "../libs/canvas_utility";



const evaluateTimeState = atom({
  key: "evaluateTimeState",
  default: 5000,
});

const detectionThresholdState = atom({
  key: "detectionThresholdState",
  default: 0.6,
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
      console.log(predictionsArr);
      let _lastChecked = get(lastCheckedState);
      let _autoDetect = get(autoDetectState);
      let _width = get(imageWidthState);
      let _height = get(imageHeightState);
      
      console.log("PIG");
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

      // This if statement prevents excessive calls to checkOverlap.
      // The rate of action dispatch is determined by the FPS, which is set by setTimeout,
      // so I cannot intervene with state updates. The value is set to 900 to allow for race conditions,
      // although it is not crucial for checkOverlap to miss a check. checkOverlap is an O(n^2) function,
      // so it is best not to spam it. Optimization in checkOverlap reduces the number of actions required.
      // The maximum input value for N is 100, so it still runs smoothly.
      let excessiveCheck = Date.now() - _lastChecked > 900 && !_autoDetect;

      if (excessiveCheck) {
        set(lastCheckedState, Date.now());
      } else {
        return;
      }



      //   //Array of ROI objects

      //If there are no selected objects the function returns because there
      //Is nothing to check
      const selectedRois = get(selectedRoi);
      if (selectedRois.length === 0) {
      return;
      }
      //If no predections have happen, then a dummy predection is sent
      //so that the function runs!
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

            //   //Array of ROI objects

      //If there are no selected objects the function returns because there
      //Is nothing to check
      // const selectedRois = get(selectedRoi);
 

      //How long it takes to evaluate if a object is there or not
      const evaluateTime = get(evaluateTimeState);
      //Get the unix time
      const currentUnixTime = Date.now();
      const selectedRoisClone = structuredClone(selectedRois);

      //   //Log N** function on quite a small scale so it's okay
      for (let index = 0; index < selectedRois.length; index++) {
        //Checking if the current
        let isOverlap = checkRectOverlap(selectedRois[index], predictionsArr);

        if (
          !roiEvaluating(
            currentUnixTime,
            selectedRois[index]["time"],
            evaluateTime
          )
        ) {
          selectedRoisClone[index]["evaluating"] = false;
        }

        if (isOverlap && selectedRois[index]["firstSeen"] === null) {
          selectedRoisClone[index]["firstSeen"] = currentUnixTime;
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;
          //If the ROI is overlapping and has "seen", see if it's lastsee - minus it's first seen is bigger than
          //the allowed time differnce if so set the occupation to true, and afterwards update the lastSeen
        } else if (isOverlap && selectedRois[index]["firstSeen"] != null) {
          let timeDiff =
            selectedRois[index]["lastSeen"] - selectedRois[index]["firstSeen"];

          if (timeDiff > evaluateTime) {
            selectedRoisClone[index].occupied = true;
          }
          selectedRoisClone[index].lastSeen = currentUnixTime;
        } else if (
          currentUnixTime - selectedRois[index]["lastSeen"] >
          evaluateTime
        ) {
          //reset the selected ROI

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
