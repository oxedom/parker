import DrawingCanvas from "./DrawingCanvas";
import CanvasInput from "./CanvasInput";

const Selector = ({selectedBoxColor,selectingBoxColor,selected,outputRef,fps,track,handleNewRoi,children}) => {

    return (<div className="cursor-crosshair pt-10">
    <DrawingCanvas
      selectedBoxColor={selectedBoxColor}
      selectingBoxColor={selectingBoxColor}
      handleNewRoi={handleNewRoi}

      selected={selected}
    >
      {children}
    </DrawingCanvas>
    <CanvasInput
      track={track}
      fps={fps}
      outputRef={outputRef}


    />
  </div>  );
}
 
export default Selector;