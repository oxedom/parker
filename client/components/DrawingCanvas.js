import { useEffect, useRef, useState } from "react";
import { renderRectangleFactory } from "../libs/canvas_utility";

const DrawingCanvas = ({imageWidth, imageHeight,children,selectedBoxColor, selectingBoxColor, handleNewRoi,}) => {


  const overlayRef = useRef(null);
  const drawCanvasRef = useRef(null);
  const [renderRectangle, setRenderRectangle] = useState(undefined);


  useEffect(() => {
    setRenderRectangle(
      renderRectangleFactory(drawCanvasRef.current, overlayRef.current)
    );
  }, [overlayRef, drawCanvasRef]);

  useEffect(() => {
    if (renderRectangle != undefined) {
      renderRectangle.setSelectedColor(selectedBoxColor);
      renderRectangle.setSelectingColor(selectingBoxColor)
    }
  }, [renderRectangle, selectedBoxColor, selectingBoxColor]);

  return (
    <>
      <canvas
        width={imageWidth}
        height={imageHeight}
        ref={drawCanvasRef}
        className="fixed"
        style={{ zIndex: 2 }}
        id="overlay"
      ></canvas>

      <canvas
        ref={overlayRef}
        width={imageWidth}
        height={imageHeight}
        onMouseDown={(e) => {
          renderRectangle && renderRectangle.handleMouseDown(e);
        }}
        onMouseMove={(e) => {
          renderRectangle && renderRectangle.handleMouseMove(e);
        }}
        onMouseOut={(e) => {
          renderRectangle && renderRectangle.handleMouseOut(e);
        }}
        onMouseUp={(e) => {
          renderRectangle && renderRectangle.handleMouseUp(e);
          let recentRoi = renderRectangle.getRecentRegion();
          handleNewRoi(recentRoi);
        }}
        className="fixed"
        style={{ zIndex: 2 }}
        id="draw_canvas"
      ></canvas>

      {children}
    </>
  );
};
export default DrawingCanvas;
