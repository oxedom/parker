import CanvasInput from "./CanvasInput";
import DrawingCanvas from "./DrawingCanvas";

const Selector = ({ track }) => {
  return (
    <div className="cursor-crosshair ">
      <DrawingCanvas></DrawingCanvas>
      <CanvasInput track={track} />
    </div>
  );
};

export default Selector;
