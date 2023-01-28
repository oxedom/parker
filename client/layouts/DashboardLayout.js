import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden h-screen bg-pink-200 ">
      <Navbar></Navbar>

      <main className="">{children}</main>
      <footer>  </footer>
    </div>
  );
};
export default DashboardLayout;
