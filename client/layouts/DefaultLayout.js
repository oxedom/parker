import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Head from "next/head";

const Layout = ({ children }) => {
  return (
    <div className="flex flex-col h-screen justify-between">
      <div className="mb-auto h-10 bg-green-500">
        <Navbar></Navbar>
        <main> {children}</main>
      </div>
      <Footer />
    </div>
  );
};
export default Layout;
