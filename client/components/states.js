import { atom, selector } from "recoil";
import uniqid from "uniqid";
import { finalName, checkRectOverlap } from "../libs/utillity";

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
        area: cords.width*cords.height,
        firstSeen: null,
        lastSeen: null,
        occupied: false,
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



    if (action.event === "occupation") {

      let {predictionsArr} = action.payload;

    //   //Array of ROI objects
      const selectedRois = get(selectedRoi);

      const selectedRoisClone = structuredClone(selectedRois);
    //   //Log N function

      for (let index = 0; index < selectedRois.length; index++) {

        let isOverlap = checkRectOverlap(selectedRois[index], predictionsArr)
        //If check that runs if a Selected ROI object is currently occupied
        //Checks that object hasn't changed occupied status by checking when it was last seen
        //and sees how long ago it was last seen, if it's under some sort of thresohold, so it will define it status
        //to unoccupied,

          //Check if 10 secounds have passed since last seen
      

           if (isOverlap && (selectedRois[index]['firstSeen'] === null)) 
          {

            selectedRoisClone[index]['firstSeen'] = Date.now();
            selectedRoisClone[index]['lastSeen'] = Date.now();
            console.log(1);
          }
          else if(isOverlap && (selectedRois[index]['firstSeen'] != null)) {
            console.log(2);
            let timeDiff = selectedRois[index]['lastSeen'] - selectedRois[index]['firstSeen']
            if(timeDiff > 5000) 
            {
              selectedRoisClone[index].occupied = true;
              selectedRoisClone[index].color = '#FFC0CB';
            }
            selectedRoisClone[index].lastSeen = Date.now();
          }
          else if(Date.now() - selectedRois[index]['lastSeen']  > 5000 ) 
          {
            //reset the selected ROI
            console.log('Pig fat');
            selectedRoisClone[index]['firstSeen'] = null
            selectedRoisClone[index]['lastSeen'] = null
            selectedRoisClone[index]['occupied'] = false
            selectedRoisClone[index]['color'] = '#0000FF'
          } 
        
          }
         

          



          console.table(selectedRoisClone[0]);
          set(selectedRoi, selectedRoisClone);
        }
      // }

    // }

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

      selectedRoisClone[targetRoiIndex] = roiClone;
      set(selectedRoi, selectedRoisClone);
    }


    }
  
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
  outputImageState,
  selectedRoiState,
};
