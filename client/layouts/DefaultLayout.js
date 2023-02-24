import Navbar from "../components/Navbar";

const DefaultLayout = ({ children }) => {
  return (
    <div className=" h-screen flex flex-col justify-between bg-green-200 ">
      <Navbar></Navbar>

      <main className=" mb-auto  ">{children}</main>
    </div>
  );
};
export default DefaultLayout;
