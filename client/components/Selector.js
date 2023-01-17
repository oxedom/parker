
import CanvasInput from "./CanvasInput";
import DrawingCanvas from "./DrawingCanvas";


const Selector = ({
  selectedBoxColor,
  selectingBoxColor,
  selected,
  outputRef,
  fps,
  track,
  handleNewRoi,
  children,
}) => {

  
  return (
    <div className="cursor-crosshair pt-10">
          <DrawingCanvas></DrawingCanvas>
      <CanvasInput track={track} fps={fps} outputRef={outputRef} />
    </div>
  );
};

export default Selector;
