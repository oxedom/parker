import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className=" h-screen flex flex-col justify-between w-max-w-[1000px] min-h-screen w-max-w-[100px]  bg-fixed bg-no-repeat bg-cover  bg-hero">
      <Navbar></Navbar>

      <main className="grow  flex flex-col justify-center items-center  ">
        {children}
      </main>
    </div>
  );
};
export default DashboardLayout;
