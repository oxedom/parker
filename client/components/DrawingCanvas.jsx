import { useEffect, useRef, useState } from "react";
import { imageHeightState, imageWidthState } from "./states";
import { useRecoilValue } from "recoil";
import { useRecoilState } from "recoil";
import { selectedRoiState } from "./states";
import { renderRoi } from "../libs/canvas_utility";
import { useWindowSize } from "../hooks/useWindowSize";

const DrawingCanvas = ({}) => {
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);
  const size = useWindowSize();
  const selectedColor = "f52222";
  const selectingColor = "#979A9A";
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);

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
  const [prevSelectedColor, setPrevSelected] = useState(null);
  const [currentCords, setCurrentCords] = useState({
    right_x: 0,
    width: 0,
    top_y: 0,
    height: 0,
  });

  //CSS style
  function grabbingCursorToogle() {
    if (isDown) {
      return "cursor-grabbing";
    } else {
      return "cursor-grab";
    }
  }

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

    const cords = {
      height: Math.abs(prevHeight),
      right_x: right_x,
      top_y: top_y,
      width: Math.abs(prevWidth),
    };

    return cords;
  }

  function addRegionOfInterest(cords) {
    //Parsing the cords to action request so it can get routed to the proper handler
    //in the selected setter

    let action = {
      event: "addRoi",
      payload: { cords },
    };
    //Sends action request with a payload, the event is handled
    //inside the state event.
    setSelectedRois(action);

    //Resetting width to 1 to prevent a nondrag click to become a square in the selectedRoi State
    setCurrentCords({
      right_x: 0,
      width: 1,
      top_y: 0,
      height: 0,
    });
  }

  function rectangleArea(rect) {
    return Math.abs(rect.width * rect.height);
  }

  //Rerender all all rois when state changes
  function updateBounding(canvasEl) {
    let canvasOffset = canvasEl.getBoundingClientRect();
    setOffsetX(canvasOffset.left);
    setOffsetY(canvasOffset.top);
  }

  useEffect(() => {
    const canvasEl = canvasRef.current;
    const overlayEl = overlayRef.current;
    updateBounding(canvasEl);
    ctxRef.current = canvasEl.getContext("2d");
    ctxoRef.current = overlayEl.getContext("2d");
    const ctx = ctxRef.current;
    const ctxo = ctxoRef.current;
    ctx.strokeStyle = selectingColor;
    ctx.lineWidth = 7;
    ctxo.strokeStyle = selectedColor;
    ctxo.lineWidth = 7;
  }, []);

  useEffect(() => {
    updateBounding(canvasRef.current);
    if (ctxoRef.current != null) {
      ctxoRef.current.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
      selectedRois.forEach((roi) => {
        renderRoi(roi, ctxoRef);
      });

      if (!isDown) {
        ctxRef.current.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );
      }
    }
  }, [selectedRois, size, imageHeight, imageWidth]);

  function handleMouseDown(e) {
    e.preventDefault();
    e.stopPropagation();

    //(0,0) Would be the top left cornor
    startX.current = parseInt(e.clientX - offsetX);
    startY.current = parseInt(e.clientY - offsetY);

    //Changes the stroke style to the selectingColor
    ctxRef.current.strokeStyle = selectingColor;
    ctxRef.current.lineWidth = 7;
    // set a flag indicating the drag has begun
    setIsDown(true);
    // setProcessing(false);
  }

  function handleMouseMove(e) {
    e.preventDefault();
    e.stopPropagation();

    // if we're not dragging, just return
    if (!isDown) {
      return;
    }

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
    ctxoRef.current.strokeStyle = prevSelectedColor;
    ctxRef.current.strokeRect(startX.current, startY.current, width, height);

    setPrevStartX(startX.current);
    setPrevStartY(startY.current);
    setPrevWidth(width);
    setPrevHeight(height);
    //Sets the mouse movements to useable cords that match the same XY graph context as the server
    //for bonding boxes

    setCurrentCords(
      convertEventCordsToRoi(prevStartX, prevStartY, prevWidth, prevHeight)
    );
  }

  function handleMouseUp(e) {
    e.preventDefault();
    e.stopPropagation();
    // Mouse dragging is over, clear the dragging flag
    setIsDown(false);
    // setProcessing(true);
    //If the area of the current cords is very small so don't add else
    if (rectangleArea(currentCords) > 500) {
      //function that handles state
      addRegionOfInterest(currentCords);
    }
  }

  function handleMouseOut(e) {
    e.preventDefault();
    e.stopPropagation();
    // the drag is over, clear the dragging flag
    setIsDown(false);
  }

  return (
    <div className={`${grabbingCursorToogle()}`}>
      {/* The canvas where a selected is drawn temporary //Cxt  */}
      <canvas
        className="fixed"
        style={{ zIndex: 2 }}
        ref={canvasRef}
        width={imageWidth}
        height={imageHeight}
      >
        <div className="absolute"></div>
      </canvas>

      {/* The canvas where all the ROIs are rendered //Cxto */}
      <canvas
        ref={overlayRef}
        width={imageWidth}
        height={imageHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseOut={handleMouseOut}
        onMouseUp={handleMouseUp}
        className="fixed"
        style={{ zIndex: 3 }}
        id="draw_canvas"
      ></canvas>
    </div>
  );
};

export default DrawingCanvas;
