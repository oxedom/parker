import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden  bg-pink-300 ">
      <Navbar></Navbar>

      <main className="">{children}</main>
    </div>
  );
};
export default DashboardLayout;
