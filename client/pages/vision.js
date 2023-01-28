import ClientRender from "../components/modes/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import ToolbarTwo from "../components/ToolbarTwo";
import { imageWidthState, imageHeightState} from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";
import Dashboard from "../components/Dashboard";


const visionPage = () => {
  const [webcamApproved, setWebCamApproved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const [totalFrames, setTotalFrames] = useState(0);
  const [processing, setProcessing] = useState(false);

  if (true) {
    return (
      <div className="flex flex-col  p-16 outline outline-1  outline-stone-900">

          {(webcamApproved && processing) ? (<Dashboard></Dashboard>) : <></> }
       
        <div>
        <div className="flex justify-between border-2 border-black">
          <RoisFeed totalFrames={totalFrames}></RoisFeed>



         

          {webcamApproved ? (
            <div className="">


 
              <DrawingCanvas></DrawingCanvas>
              <ClientRender
                loaded={loaded}
                webcamApprove={webcamApproved}
                setTotalFrames={setTotalFrames}
                setLoaded={setLoaded}
              ></ClientRender>
       
            </div>
          ) : (
            <img width={imageWidth}  height={imageHeight}   style={{ zIndex: 1 }}/>  
)}




        
            <ToolbarTwo
              webcamApproved={webcamApproved}
              setWebCamApproved={setWebCamApproved}
              setProcessing={setProcessing}
              processing={processing}
            >
              {" "}
            </ToolbarTwo>
            </div>
        </div>
      </div>
    );
  }
};

export default visionPage;
