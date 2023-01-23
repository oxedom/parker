import Head from "next/head";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar"

const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden" >

    
        <div className=" flex ">
        <Sidebar></Sidebar>
        <main> {children}</main>
        </div>
    </div>
  );
}
export default DashboardLayout;
