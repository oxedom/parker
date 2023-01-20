import Head from "next/head";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar"
const DashboardLayout = ({ children }) => {
  return (
    <div className="overflow-hidden" >

    
        <div className="bg-blue-100 flex p-28">
        <Sidebar></Sidebar>
        <main> {children}</main>
        </div>
    </div>
  );
};
export default DashboardLayout;
