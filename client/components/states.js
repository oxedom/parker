import { atom, selector } from "recoil";

// const charState = selector({
//     key: 'charState',
//     get: ({get}) => {
//         const name = get(nameState)
//         return name.length ;
//     }
// })

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

export { imageWidthState, imageHeightState, selectingColorState,selectedColorColorState};
