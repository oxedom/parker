import { atom, selector } from "recoil";
import uniqid from "uniqid";
import { finalName } from "../libs/utillity";

const selectedRoi= atom({
  key: "selectedRois",
  default:[],
});

const roiTypeState= atom({
  key: "roiType",
  default:"Any",
});

const roiNameState= atom({
  key: "roiName",
  default:"",
});


const selectedRoiState = selector({
  key: 'selectedRoisState',
  default:[],
  get: ({ get }) => {
    const selectedRois = get(selectedRoi)
    return selectedRois
  },

  set: ({ set, get}, action) => {


    if(action.event === 'addRoi')
    {
      let cords = action.payload
      let date = new Date();
      let roiType = get(roiTypeState)
      let roiName = get(roiNameState)
      const oldRois = get(selectedRoi)
  
      const roiObj = {
        name: finalName(roiName, oldRois.length),
        roi_type: roiType,
        cords: { ...cords },
        time: date.getTime(),
        uid: uniqid(),
      };
  
      const updatedArr = [...oldRois, roiObj]
  
      set(selectedRoi,updatedArr) 

    }
   if(action.event === 'deleteRoi') 
   {
    let uid = action.payload
    const oldRois = get(selectedRoi)
    const updatedArr = oldRois.filter(roi => roi.uid !== uid)
    set(selectedRoi, updatedArr)
   }

  },





});


const selectingColorState = atom({
  key: "selectingColor",
  default:"#EC3945",
});

const selectedColorColorState = atom({
  key: "selectedColor",
  default:"#8FC93A",
});




const imageHeightState = atom({
  key: "imageHeight",
  default: 720,
});

const imageWidthState = atom({
  key: "imageWidth",
  default: 1280,
});

export { 
  roiTypeState,
  roiNameState,
  imageWidthState,
   imageHeightState,
    selectingColorState,
    selectedColorColorState,
     selectedRoiState};
