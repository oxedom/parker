import { useRecoilValue } from "recoil";
import { outputImageState} from "../components/states";


const ProcessingFeed = () => {

    const outputImage = useRecoilValue(outputImageState)


    return ( <div>

        <img src={outputImage} />
    </div> );
}
 
export default ProcessingFeed;