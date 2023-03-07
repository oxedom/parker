import { atom, selector } from "recoil";
import uniqid from "uniqid";
import { checkRectOverlap, selectedFactory } from "../libs/utillity";
import { roiEvaluating } from "../libs/states_utility";
import { clearCanvas, renderAllOverlaps } from "../libs/canvas_utility";

const evaluateTimeState = atom({
  key: "evaluateTimeState",
  default: 5000,
});

const detectionThresholdState = atom({
  key: "detectionThresholdState",
  default: 0.6,
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

const lastCheckedState = atom({
  key: "lastChecked",
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
      const roiObj = selectedFactory(cords)

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
      let _selectedRoi = get(selectedRoi);
      let _autoDetect = get(autoDetectState);
      let _width = get(imageWidthState);
      let _height = get(imageHeightState);

      if (_autoDetect) {
        
      
        let updatedArr = []
        predictionsArr.forEach(pred => 
          {
            let roiObj = selectedFactory(pred.cords)

              updatedArr.push(roiObj)
            
    
          })
          set(selectedRoi, updatedArr)

  
        set(autoDetectState, false);
      }

      if (get(showDetectionsState) && predictionsArr.length > 0) {
        renderAllOverlaps(predictionsArr, canvas, _width, _height);
      } else {
        clearCanvas(canvas, _width, _height);
      }

      //This if checks prevents the checkOverlap running an execive amount of times,
      //The rate of the action dispatched is depent on the rate of FPS, and because the FPS is
      //determened by a setTimeout I can't internvine with state because the call stack values are
      //updated up untill the point where the timer was inited.
      //This is set too 900 to give some leeway for race condtions, altohugh it's not very crucial that checkOverlap will
      //Miss one check up. As well as checkOverlap is a O^n2 function so best not to spam it, could be done better but because
      //max input of N would at it's highest point 100 (And that an exaggerating  it still would run smoothly (tested,
      // as well there //Is some optimazation in the checkOverlap which would make the amount of actions less.
      let excessiveCheck = Date.now() - _lastChecked > 900 && !_autoDetect;

      if (excessiveCheck) {
        set(lastCheckedState, Date.now());
      } else {
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
      const selectedRois = get(selectedRoi);
      if (selectedRois.length === 0) {
        return;
      }
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

    if (action.event === "addManyRois") {
      let date = new Date();

      let { predictionsArr } = action.payload;
      let rois = [];
      predictionsArr.forEach((p) => {
        const roiObj = {
          label: "vehicle",
          cords: { ...p.cords },
          time: date,
          uid: uniqid(),
          area: p.width * p.height,
          firstSeen: null,
          lastSeen: null,
          occupied: null,
          hover: false,
          evaluating: true,
        };
        rois.push(roiObj);
      });

      const oldRois = get(selectedRoi);

      const updatedArr = [...oldRois, ...rois];

      set(selectedRoi, updatedArr);
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
  autoDetectState,
};
