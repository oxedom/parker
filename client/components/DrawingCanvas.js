import { useEffect, useRef, useState } from "react";
import { renderRectangleFactory } from "../libs/canvas_utility";
import uniqid from 'uniqid';
const DrawingCanvas = ({ imageWidth, imageHeight, children, setSelected, selected}) => {
  const overlayRef = useRef(null);
  const drawCanvasRef = useRef(null);
  const [renderRectangle, setRenderRectangle] = useState(undefined);
  const SSR = typeof window === "undefined";

  useEffect(() => {
    setRenderRectangle(
      renderRectangleFactory(drawCanvasRef.current, overlayRef.current)
    );
  }, [overlayRef, drawCanvasRef]);

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
          renderRectangle.handleMouseDown(e);
        }}
        onMouseMove={(e) => {
          renderRectangle.handleMouseMove(e);
        }}
        onMouseOut={(e) => {
      
          renderRectangle.handleMouseOut(e);
        }}
        onMouseUp={(e) => {
          renderRectangle.handleMouseUp(e);
          setSelected(renderRectangle.getSelectedRegions())
          console.log(selected);
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
