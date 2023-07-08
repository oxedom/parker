import { atom, selector } from "recoil";
import { checkRectOverlap, selectedFactory } from "../libs/utillity";
import {
  checkRoiEvaluating,
  overlapsAndKnown,
  firstDetect,
  calculateTimeDiff,
  supressedRoisProcess,
  convertSuppressedRoisToSelected,
  SnapshotFactory,
} from "../libs/states_utility";
import { renderAllOverlaps, drawTextOnCanvas } from "../libs/canvas_utility";

//The evaluation time is used to have a minimum time a square needs to be occupied/unoccupied to change it's state.
const evaluateTimeState = atom({
  key: "evaluateTimeState",
  default: 5000,
});

//Automatic Evaluate time is 10 secounds
const autoEvaluateTimeState = atom({
  key: "autoEvaluateTimeState",
  default: 10000,
});

//Default thershold for a detection
const detectionThresholdState = atom({
  key: "detectionThresholdState",
  default: 0.5,
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
  default: 0.2,
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
    //Adds Roi to array
    if (action.event === "addRoi") {
      let { cords } = action.payload;
      const roiObj = selectedFactory(cords);

      const oldRois = get(selectedRoi);

      const updatedArr = [...oldRois, roiObj];

      set(selectedRoi, updatedArr);
    }
    //Removes ROI from array
    if (action.event === "deleteRoi") {
      let uid = action.payload;
      const oldRois = get(selectedRoi);
      const updatedArr = oldRois.filter((roi) => roi.uid !== uid);
      set(selectedRoi, updatedArr);
    }

    if (action.event === "occupation") {
      // console.time('occupation');
      // let start = performance.now();
      let { predictionsArr, canvas } = action.payload;

      let _lastChecked = get(lastCheckedState);
      let _autoDetect = get(autoDetectState);
      let _width = get(imageWidthState);
      let _height = get(imageHeightState);
      const currentUnixTime = Date.now();

      //
      if (get(showDetectionsState) && predictionsArr.length > 0) {
        renderAllOverlaps(predictionsArr, canvas, _width, _height);
      }

      // This if statement prevents excessive calls to checkOverlap with ROIS.
      //   let excessiveCheck = ((Date.now() - _lastChecked > 900) && !_autoDetect)

      //   if (excessiveCheck) {

      //     set(lastCheckedState, Date.now());
      //   } else if (!_autoDetect) {
      //  ;
      //     return ;
      //   }

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

        //If true check if still evaluting
        if (selectedRoisClone[index]["evaluating"]) {
          let roiStillEvaluating = checkRoiEvaluating(
            currentUnixTime,
            selectedRois[index]["events"][0]["timeMarked"],
            evaluateTime
          );

          if (!roiStillEvaluating) {
            selectedRoisClone[index]["evaluating"] = false;
          }
        }

        //Runs if there is no overlap and it's the firt time it's been seen (Null first seen and overlap)
        //Set the time trackings to the same time
        if (firstDetect(isOverlap, selectedRois, index)) {
          selectedRoisClone[index]["firstSeen"] = currentUnixTime;
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;

          //Runs if there is an overlap (Thats passed the threshold) and it's been seen
        } else if (overlapsAndKnown(isOverlap, selectedRois, index)) {
          //Setting last seen to unix time to start/continue tracking the state of the parking lot
          selectedRoisClone[index]["lastSeen"] = currentUnixTime;

          //Calculate how long the item has been tracked for
          let timeDiff = calculateTimeDiff(selectedRois, index);

          //If the amount of time (timeDiff) the region has been tracked passes the evalute time
          //threshold update that state of the region to be occupied
          if (timeDiff > evaluateTime) {
            if (!selectedRoisClone[index]["occupied"]) {
              selectedRoisClone[index].occupied = true;
              selectedRoisClone[index].cycleCount += 1;
              let occupiedEvent = {
                cycle: selectedRoisClone[index]["cycleCount"],
                eventName: "occupied",
                timeMarked: currentUnixTime,
                duration: null,
              };
              selectedRoisClone[index]["events"].push(occupiedEvent);
            }
            let currentCycleCount = selectedRoisClone[index]["cycleCount"];
            selectedRoisClone[index]["events"][currentCycleCount]["duration"] =
              selectedRoisClone[index].lastSeen -
              selectedRoisClone[index].firstSeen;

            //Keep track of the of the event duration and keeping occupied true
          }
          //Runs if there is no overlap and checks if the region hasn't been seen for evaluate time.
          //If so it peforms a reset of the tracking
        } else if (
          currentUnixTime - selectedRois[index]["lastSeen"] >
          evaluateTime
        ) {
          //Reset tracking
          if (selectedRoisClone[index]["occupied"]) {
            selectedRoisClone[index]["firstSeen"] = null;
            selectedRoisClone[index]["lastSeen"] = null;
            selectedRoisClone[index]["occupied"] = false;
          }
        }
      }

      //Fires while auto Detectection mode is processing
      if (_autoDetect) {
        const autoChecked = get(autoCheckedState);
        const autoEvaluateTime = get(autoEvaluateTimeState);
        drawTextOnCanvas(canvas, _width, _height, "Auto detecting");
        let adding = Date.now() - autoChecked;
        if (autoChecked === 0) {
          set(autoCheckedState, Date.now());
        } else if (adding <= autoEvaluateTime) {
          const autoDetectArr = get(autoDetectArrState);
          set(autoDetectArrState, [...autoDetectArr, predictionsArr]);
        } else {
          const autoDetectArr = get(autoDetectArrState);

          //Const percent that it need to be in the detectionsArr
          const minimumAttendance = 0.6;
          const suppresedRois = supressedRoisProcess(
            autoDetectArr,
            minimumAttendance
          );

          const convertedToSelected =   convertSuppressedRoisToSelected(suppresedRois);

          set(selectedRoi, convertedToSelected);
          set(autoDetectState, false);
          set(autoDetectArrState, []);
          set(autoCheckedState, 0);
        }
      } else {

        set(selectedRoi, selectedRoisClone);
      }
      let stop = performance.now();
      // console.log("someFunc took " + (stop - start) + " milliseconds");
      // console.timeEnd('occupation');
    }
    //When a ROI is hovered over it updates that state that it's hover property is true, which in turn effects it's what color it is renderered
    //in the ROI FEED and on the canvas
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
    if (action.event === "importSelected") {
      let selectionData = localStorage.getItem("selections");
      let parsed = JSON.parse(selectionData);
      set(selectedRoi, parsed.selectedRegions);
    }
    //When a ROI is unhovered over it updates that state that it's hover property is no longer true
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

//Default Width and Heights
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
