import { atom, selector } from "recoil";
import uniqid from "uniqid";
import { finalName, checkRectOverlap } from "../libs/utillity";
import { roiEvaluating } from "../libs/states_utility";

const selectedRoi = atom({
  key: "selectedRois",
  default: [],
});

const evaluateTimeState = atom({
  key: "evaluateTimeState",
  default: 5000,
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
  default: true,
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
        cords: { ...cords },
        time: date.getTime(),
        uid: uniqid(),
        area: cords.width * cords.height,
        firstSeen: null,
        lastSeen: null,
        occupied: false,
        hover: false,
        evaluating: true
      };
      console.log(roiObj);
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
      let { predictionsArr } = action.payload;

      //   //Array of ROI objects
      const selectedRois = get(selectedRoi);
      const evaluateTime = get(evaluateTimeState);
      const selectedRoisClone = structuredClone(selectedRois);
      //   //Log N function

      for (let index = 0; index < selectedRois.length; index++) {
        let isOverlap = checkRectOverlap(selectedRois[index], predictionsArr);
        //If check that runs if a Selected ROI object is currently occupied
        //Checks that object hasn't changed occupied status by checking when it was last seen
        //and sees how long ago it was last seen, if it's under some sort of thresohold, so it will define it status
        //to unoccupied,

        //Check if 10 secounds have passed since last seen
        const currentUnixTime = Date.now()
      
        //If the ROI is overlapping and hasn't been "seen" set it's timestames to now
        if(!roiEvaluating(currentUnixTime, selectedRois[index]['time'], evaluateTime)) 
        {
          selectedRoisClone[index]['evaluating'] = false
        }
        if (isOverlap && selectedRois[index]["firstSeen"] === null) {

          selectedRoisClone[index]["firstSeen"] = currentUnixTime
          selectedRoisClone[index]["lastSeen"] = currentUnixTime
      //If the ROI is overlapping and has "seen", see if it's lastsee - minus it's first seen is bigger than
      //the allowed time differnce if so set the occupation to true, and afterwards update the lastSeen 
        } else if (isOverlap && selectedRois[index]["firstSeen"] != null) {

          let timeDiff =
            selectedRois[index]["lastSeen"] - selectedRois[index]["firstSeen"];

          if (timeDiff > evaluateTime) {
            selectedRoisClone[index].occupied = true;
          }
          selectedRoisClone[index].lastSeen = currentUnixTime
        } 

        

        else if (
          
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

      //Toogle color to selected blue
      roiClone.color = "#0073ff";
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

      //Toogle color to selected blue
      roiClone.color = "#FF0000";
      roiClone.hover = false;

      selectedRoisClone[targetRoiIndex] = roiClone;
      set(selectedRoi, selectedRoisClone);
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
  default: "#f52222",
});

const detectionColorState = atom({
  key: "detectionColor",
  default: "#0000FF",
});

const imageHeightState = atom({
  key: "imageHeight",
  default: 1,
});

const imageWidthState = atom({
  key: "imageWidth",
  default: 1,
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
  evaluateTimeState,
  outputImageState,
  selectedRoiState,
};
