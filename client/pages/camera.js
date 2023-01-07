import Layout from "../layouts/DefaultLayout";
import Webcam from "../components/Webcam";
import Canvas from "../components/Canvas";
import {  useRef } from 'react';



const Camera = () => {



    



    



  
    // renderRectangle.printHello();

    return ( 
    <Layout>
    <div> 
        {/* MAKE THEM BOTH ABSOULTE AND REMEMBER TO CLONE THE IMAGE FROM THE WEBCAM TO A CANVAS IMAGE */}
        {/* IT'S NOT IN THE RENDER RECTANGLE FACTORY, ALSO MAKE REFS WITH REACT AND SEND T HEM TO THE components */}

        <h4>Welcome to Camera page</h4> 
        <Canvas></Canvas>
        <Webcam></Webcam>

        
        
        </div>
</Layout>);
}
 
export default Camera;