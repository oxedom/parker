
import CanvasInput from "./CanvasInput";
import DrawingCanvas from "./DrawingCanvas";


const Selector = ({
  outputRef,
  fps,
  track,
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
