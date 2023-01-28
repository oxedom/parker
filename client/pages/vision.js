import ClientRender from "../components/modes/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import ToolbarTwo from "../components/ToolbarTwo";

const visionPage = () => {
  const [webcamApproved, setWebCamApproved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [totalFrames, setTotalFrames] = useState(0);
  const [processing, setProcessing] = useState(false);

  if (true) {
    return (
      <div className="flex  p-16 outline outline-1  outline-stone-900">
        {/* <Toolbar webcamApprove={webcamApproved} setWebCamApprove={setWebCamApproved}></Toolbar> */}

        <div className="flex border-2 border-black">
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
            <video className=""  style={{ zIndex: 1 }} width={1280} height={720}> </video>
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
    );
  }
};

export default visionPage;
