import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden  bg-pink-300">
      <Navbar></Navbar>
      <div className=" flex flex-col  mx-auto my-10 px-48  ">
        <div className=" flex flex-col sm:flex-row">
          <main className="flex">{children}</main>
        </div>
      </div>
    </div>
  );
};
export default DashboardLayout;
