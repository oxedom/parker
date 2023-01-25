import Camera from "../../components/Camera";
import Toolbar from "../../components/Toolbar";
import RoisFeed from "../../components/RoisFeed";

const ParkingPage = () => {
  return (
    <>
      <div className="flex m-2  ">
        <Toolbar></Toolbar>
        <Camera></Camera>
        <RoisFeed></RoisFeed>
      </div>
    </>
  );
};

export default ParkingPage;
