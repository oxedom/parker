import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Head from "next/head";

const Layout = ({ children }) => {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar></Navbar>
      <main> {children}</main>
      <Footer />
    </div>
  );
};
export default Layout;
