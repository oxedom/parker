import { useEffect, useRef,useState} from "react";
import {renderRectangleFactory, } from "../libs/canvas_utility";
const DrawingCanvas = ({imageWidth, imageHeight}) => {

    // const overlayRef = useRef(null)
    // const drawCanvasRef = useRef(null)
    const [renderRectangle, setRenderRectangle] = useState(undefined)
    const SSR = typeof window === 'undefined'
    

    useEffect(() => 
    {
        if(!SSR) 
        {
         console.log('CLIENT');
        const drawCanvasRef = document.getElementById('draw_canvas')
        const overlayRef = document.getElementById('overlay')
        setRenderRectangle(renderRectangleFactory(drawCanvasRef, overlayRef))

        }
   
    },[])


    return (
    <div className="bg-blue-500">
    <canvas width={imageWidth} height={imageHeight} 
    className="absolute" id="overlay"></canvas> 
    
    <canvas
    width={imageWidth} height={imageHeight}
    onMouseDown={(e) => { renderRectangle.handleMouseDown(e) }}
    onMouseMove={(e) => {  renderRectangle.handleMouseMove(e) }}
    onMouseOut={(e) => { renderRectangle.handleMouseOut(e) }}
    onMouseUp={(e) => { renderRectangle.handleMouseUp(e) }}
    className="absolute" id="draw_canvas">
    </canvas>
    </div> );

    }
export default DrawingCanvas;