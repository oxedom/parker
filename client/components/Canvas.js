import { useEffect, useRef } from 'react';
import {renderRectangleFactory} from '../libs'

const Canvas = () => {



    const overlayRef = useRef(null)
    const canvasRef = useRef(null)
    useEffect(()=>{
        const renderRectangle = renderRectangleFactory(canvasRef.current,overlayRef.current)


    },[])



    return ( <>

    <canvas ref={overlayRef}></canvas>
    <canvas ref={canvasRef} ></canvas>

    </> );
}
 
export default Canvas;