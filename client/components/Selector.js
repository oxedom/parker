import DrawingCanvas from "./DrawingCanvas";
import CanvasInput from "./CanvasInput";


const Selector = ({selectedBoxColor,selectingBoxColor,imageHeight,imageWidth,selected,outputRef,fps,track,handleNewRoi,children}) => {

    return (<div className="cursor-crosshair pt-10">
    <DrawingCanvas
      selectedBoxColor={selectedBoxColor}
      selectingBoxColor={selectingBoxColor}
      handleNewRoi={handleNewRoi}
      imageWidth={imageWidth}
      imageHeight={imageHeight}
      selected={selected}
    >
      {children}
    </DrawingCanvas>
    <CanvasInput
      track={track}
      fps={fps}
      outputRef={outputRef}
      imageWidth={imageWidth}
      imageHeight={imageHeight}
    />
  </div>  );
}
 
export default Selector;