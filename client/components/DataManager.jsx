import Button from "./Button";
import { useRecoilValue,  } from "recoil";
import {  selectedRoiState } from "./states";
const DataManger = () => {

    const selectedRois = useRecoilValue(selectedRoiState);



    function handleJson() {

        const beforeExport = 
        {
            title: 'Parkerr.org',
            repo: 'https://github.com/oxedom/parker',
            timeOfExport: Date.now(),
            rois: selectedRois
        }

  // Convert the data object to JSON
     const jsonData = JSON.stringify(beforeExport);
  // Create a Blob with the JSON data
     const blob = new Blob([jsonData], { type: 'application/json' });
  // Create a temporary URL for the Blob
     const url = URL.createObjectURL(blob);
  // Create a link element
    const link = document.createElement('a');
    link.href = url;
    link.download = 'exported_data.json';
  // Simulate a click on the link element to trigger the file download
    link.click();
  // Clean up the temporary URL
    URL.revokeObjectURL(url);
    }
    function handleCsv() {}


    return ( <>

        <div className="bg-black/60 backdrop-blur-sm pb-3 px-5  flex flex-col max-w-[500px]" >
        <h2 className="  text-center text-3xl text-white pt-2 border-b border-orange-500 pb-1" > Data manger </h2>
        <p className="text-white text-xl pt-4 "> Selection boxes keep track of events that occur, export that data 
        locally to perform your own data exploration.
        </p>

        <div className="flex flex-col items-center mt-10 gap-y-2">
        <h1 className="text-white text-2xl "> Export as </h1>
            <div className="flex justify-center items-center gap-5 ">
           
            <Button onClick={handleJson}> <span className="text-lg"> JSON</span> </Button>
            {/* <Button> <span className="text-lg"> CSV </span> </Button> */}
            </div>
        </div>


        </div>



        </>

     );
}
 
export default DataManger;