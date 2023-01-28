import Head from "next/head";

import Navbar from "../components/Navbar";

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden  bg-pink-300">
      <Navbar></Navbar>

        {/* <div className="flex flex-col  mx-auto my-10 px-48 pr-48"> */}
          <main className="">{children}</main>
        {/* </div> */}

    </div>
  );
};
export default DashboardLayout;
