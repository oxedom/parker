import { useEffect, useRef, useState } from "react";
import { selectedColorColorState, selectingColorState } from "./states";
import { useRecoilValue } from "recoil"
const DrawingCanvas = () => {


 
  const selectedRegions = [];
  const selectedColor = useRecoilValue(selectedColorColorState);
  const selectingColor= useRecoilValue(selectingColorState);


  let ctxRef = useRef(null);
  let ctxoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const startX = useRef(null);
  const startY = useRef(null);

  const [isDown, setIsDown] = useState(false);
  const [offsetX, setOffsetX] = useState(undefined);
  const [offsetY, setOffsetY] = useState(undefined);
  const [prevStartX, setPrevStartX] = useState(0);
  const [prevStartY, setPrevStartY] = useState(0);
  const [prevWidth, setPrevWidth] = useState(0);
  const [prevHeight, setPrevHeight] = useState(0);
  const [prevSelectedColor, setPrevSelected] = useState(null)

  function convertEventCordsToRoi(
    prevStartX,
    prevStartY,
    prevWidth,
    prevHeight
  ) {
    let right_x = null;
    let top_y = null;

    prevWidth < 0
      ? (right_x = prevStartX - Math.abs(prevWidth))
      : (right_x = prevStartX);
    prevHeight < 0
      ? (top_y = prevStartY - Math.abs(prevHeight))
      : (top_y = prevStartY);

    let date = new Date();
    const cords = {
      height: Math.abs(prevHeight),
      right_x: right_x,
      top_y: top_y,
      width: Math.abs(prevWidth),
    };

    return cords;
  }

  function _addRegionOfIntrest(cords) {
    let date = new Date();
    const roiObj = {
      cords: { ...cords },
      time: date.getTime(),
    };

    selectedRegions.push(roiObj);

    return selectedRegions;
  }

  function rectangleArea(rect) {
    return Math.abs(rect.width * rect.height);
  }

  useEffect(() => {
    const canvasEl = canvasRef.current;
    const overlayEl = overlayRef.current;
    let canvasOffset = canvasEl.getBoundingClientRect();
    setPrevSelected(selectedColor)
    ctxRef.current = canvasEl.getContext("2d");
    ctxoRef.current = overlayEl.getContext("2d");

    const ctx = ctxRef.current;
    const ctxo = ctxoRef.current;
    ctx.strokeStyle = selectingColor;
    ctx.lineWidth = 10;
    ctxo.strokeStyle = selectedColor;
    ctxo.lineWidth = 10;
    setOffsetX(canvasOffset.left);
    setOffsetY(canvasOffset.top);
  }, []);

  function handleMouseDown(e) {
    e.preventDefault();
    e.stopPropagation();

    //(0,0) Would be the top left cornor
    //(Max Width, Max Height ) woul be the bottom right cornor
    // save the starting x/y of the rectangle
    startX.current = parseInt(e.clientX - offsetX);
    startY.current = parseInt(e.clientY - offsetY);

    // set a flag indicating the drag has begun
    setIsDown(true);
  }

  function handleMouseMove(e) {
    e.preventDefault();
    e.stopPropagation();

    // if we're not dragging, just return
    if (!isDown) {
      return;
    }

    // get the current mouse position
    //(0,0) Would be the top left cornor
    //(Max Width, Max Height ) woul be the bottom right cornor
    // save the starting x/y of the rectangle
    const mouseX = parseInt(e.clientX - offsetX);
    const mouseY = parseInt(e.clientY - offsetY);

    // calculate the rectangle width/height based
    // on starting vs current mouse position
    var width = mouseX - startX.current;
    var height = mouseY - startY.current;

    // clear the canvas
    ctxRef.current.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    // draw a new rect from the start position
    // to the current mouse position

    ctxRef.current.strokeRect(startX.current, startY.current, width, height);

    setPrevStartX(startX.current);
    setPrevStartY(startY.current);
    setPrevWidth(width);
    setPrevHeight(height);
  }

  function handleMouseUp(e) {
    e.preventDefault();
    e.stopPropagation();

    // the drag is over, clear the dragging flag
    setIsDown(false);

    // ctxo.strokeRect(random.left_x, random.top_y, random.width, random.height);
    ctxRef.current.strokeStyle = selectingColor;
    ctxRef.current.lineWidth = 10;
    ctxoRef.current.strokeStyle = prevSelectedColor;
    ctxoRef.current.lineWidth = 10;
    setPrevSelected(selectedColor)
    
    let cords = convertEventCordsToRoi(
      prevStartX,
      prevStartY,
      prevWidth,
      prevHeight
    );
    if (rectangleArea(cords) < 500) {
      return;
    } else {
      ctxoRef.current.strokeRect(prevStartX, prevStartY, prevWidth, prevHeight);
    
      _addRegionOfIntrest(cords);
    }
  }

  function handleMouseOut(e) {
    e.preventDefault();
    e.stopPropagation();

    // the drag is over, clear the dragging flag
    setIsDown(false);
  }

  return (
    <div>
      <canvas
        className="fixed"
        style={{ zIndex: 2 }}
        ref={canvasRef}
        width={1280}
        height={720}
      ></canvas>

      <canvas
        ref={overlayRef}
        width={1280}
        height={720}
        onMouseDown={(e) => {
          handleMouseDown(e);
        }}
        onMouseMove={(e) => {
          handleMouseMove(e);
        }}
        onMouseOut={(e) => {
          handleMouseOut(e);
        }}
        onMouseUp={(e) => {
          handleMouseUp(e);
        }}
        className="fixed"
        style={{ zIndex: 3 }}
        id="draw_canvas"
      ></canvas>
    </div>
  );
};

export default DrawingCanvas;
