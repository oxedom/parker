import Navbar from "../components/Navbar";

const DefaultLayout = ({ children }) => {
  return (
    <div className="h-screen flex flex-col justify-between min-h-screen  bg-fixed bg-no-repeat bg-cover  bg-hero  ">
      <Navbar></Navbar>

      <main className=" mb-auto  ">{children}</main>
    </div>
  );
};
export default DefaultLayout;
