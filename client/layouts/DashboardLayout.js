import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden h-screen flex flex-col justify-between bg-pink-200 ">
      <Navbar></Navbar>

      <main className="grow mb-auto flex flex-col justify-center items-center">
        {children}
      </main>
      <footer className="bg-green-400"> I am footer </footer>
    </div>
  );
};
export default DashboardLayout;
