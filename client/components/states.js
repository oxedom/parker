import { atom, selector } from "recoil";

// const charState = selector({
//     key: 'charState',
//     get: ({get}) => {
//         const name = get(nameState)
//         return name.length ;
//     }
// })
const selectedRoi= atom({
  key: "selectedRois",
  default:[],
});



const selectedRoiState = selector({
  key: 'selectedRoisState',
  default:0,
  get: ({ get }) => {
    const selectedRois = get(selectedRoi)
    return selectedRois
  },
  set: ({ set, get}, cords) => {

    let date = new Date();

    const roiObj = {
      cords: { ...cords },
      time: date.getTime(),
    };
    
  
    const oldRois = get(selectedRoi)

    const updatedArr = [...oldRois, roiObj]

    set(selectedRoi,updatedArr) 

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

export { imageWidthState, imageHeightState, selectingColorState,selectedColorColorState, selectedRoiState};
