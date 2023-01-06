import Layout from "../layouts/DefaultLayout";
import Webcam from "../components/Webcam";
import {renderRectangleFactory} from '../libs'



const Camera = () => {


    const renderRectangle = renderRectangleFactory();
    renderRectangle.printHello();

    return ( 
    <Layout>
    <div> 
        {/* MAKE THEM BOTH ABSOULTE AND REMEMBER TO CLONE THE IMAGE FROM THE WEBCAM TO A CANVAS IMAGE */}
        {/* IT'S NOT IN THE RENDER RECTANGLE FACTORY, ALSO MAKE REFS WITH REACT AND SEND T HEM TO THE components */}

        <canvas className=""></canvas>
        <canvas></canvas>
        <h4>Welcome to Camera page</h4> 
        <Webcam></Webcam>
        
        
        
        </div>
</Layout>);
}
 
export default Camera;