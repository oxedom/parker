import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden">
      <Navbar></Navbar>
      <div className=" flex flex-col  mx-auto my-10 px-48">
        <div className=" flex flex-col sm:flex-row">
          <main className="flex bg-indigo-100">{children}</main>
        </div>
      </div>
    </div>
  );
};
export default DashboardLayout;
