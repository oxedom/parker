import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className=" h-screen flex flex-col justify-between bg-blue-200 ">
      <Navbar></Navbar>

      <main 
      className="grow mb-auto flex flex-col justify-center items-center  "
      >
        
        {children}
      </main>
    </div>
  );
};
export default DashboardLayout;
