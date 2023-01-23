import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden">
      <Navbar></Navbar>
      <div className=" flex flex-col  mx-auto my-10 px-48">
      <div class="flex-1 flex flex-col sm:flex-row">
      <main class="flex-1 bg-indigo-100">{children}</main>
    </div>
      </div>
    </div>
  );
};
export default DashboardLayout;
