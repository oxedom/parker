import Camera from "../../components/Camera";
import Toolbar from "../../components/Toolbar";
import RoisFeed from "../../components/RoisFeed";

const ParkingPage = () => {
  return <>

    <div className="flex justify-center m-2 border-solid border-4 border-violet-500">
    <Toolbar></Toolbar>
    <Camera></Camera>
    <RoisFeed></RoisFeed>
    </div>

  </>;
};

export default ParkingPage;
