
async function bufferToServer(capturedImage) {
    const imagePhoto = await capturedImage.takePhoto();
    let imageBuffer = await imagePhoto.arrayBuffer();
    imageBuffer = new Uint8Array(imageBuffer);
  
    const res = await fetch(flask_url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ buffer: [...imageBuffer] }),
    });
  
    const data = await res.json();
  
    return data;
  }

function renderRectangleFactory(canvasRef,overlayRef) {

    const selectedRegions = [];

    const ctx = canvasRef.getContext("2d");
    const ctxo = overlayRef.getContext("2d");

    ctx.strokeStyle = "blue";
    ctx.lineWidth = 10;
    ctxo.strokeStyle = "blue";
    ctxo.lineWidth = 4;
    let canvasOffset = canvasRef.getBoundingClientRect();
    let offsetX = canvasOffset.left;
    let offsetY = canvasOffset.top;
    // this flage is true when the user is dragging the mouse
    let isDown = false;
    // these vars will hold the starting mouse position
    let startX;
    let startY;
  
    let prevStartX = 0;
    let prevStartY = 0;
  
    let prevWidth = 0;
    let prevHeight = 0;
  
    function handleMouseDown(e) {
      e.preventDefault();
      e.stopPropagation();
  
      // save the starting x/y of the rectangle
      startX = parseInt(e.clientX - offsetX);
      startY = parseInt(e.clientY - offsetY);
  
      // set a flag indicating the drag has begun
      isDown = true;
    }
  
    function handleMouseUp(e) {
      e.preventDefault();
      e.stopPropagation();
  
      // the drag is over, clear the dragging flag
      isDown = false;
      // ctxo.strokeRect(random.left_x, random.top_y, random.width, random.height);
      ctxo.strokeRect(prevStartX, prevStartY, prevWidth, prevHeight);
  
      _addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight);
  
    }
  
    function handleMouseOut(e) {
      e.preventDefault();
      e.stopPropagation();
  
      // the drag is over, clear the dragging flag
      isDown = false;
    }
  
    function handleMouseMove(e) {
      e.preventDefault();
      e.stopPropagation();
  
      // if we're not dragging, just return
      if (!isDown) {
        return;
      }
  
      // get the current mouse position
      const mouseX = parseInt(e.clientX - offsetX);
      const mouseY = parseInt(e.clientY - offsetY);
  
      // Put your mousemove stuff here
  
      // calculate the rectangle width/height based
      // on starting vs current mouse position
      var width = mouseX - startX;
      var height = mouseY - startY;
  
      // clear the canvas
      ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
  
      // draw a new rect from the start position
      // to the current mouse position
      ctx.strokeRect(startX, startY, width, height);
  
      prevStartX = startX;
      prevStartY = startY;
  
      prevWidth = width;
      prevHeight = height;
  
      
    }
  
    function _addRegionOfIntrest(prevStartX, prevStartY, prevWidth, prevHeight) {
      const roiObj = {
        label: selectedType,
        cords: {
          right_x: prevStartX,
          top_y: prevHeight + prevStartY,
          left_x: prevWidth + prevStartX,
          bottom_y: prevStartY,
        },
      };
    
      selectedRegions.push(roiObj);
      return selectedRegions;
    }
  
  
  
    function getSelectedRegions() 
    {
      return selectedRegions
    }
  
  
    // function printHello() 
    // {}
  
  
    return { handleMouseDown, handleMouseMove, handleMouseUp, handleMouseOut, getSelectedRegions };
  }
  

  export { renderRectangleFactory, bufferToServer}