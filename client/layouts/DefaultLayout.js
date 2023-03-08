import Navbar from "../components/Navbar";

const DefaultLayout = ({ children }) => {
  return (
    <div className="h-screen flex flex-col justify-between min-h-screen  bg-fixed bg-no-repeat bg-cover     ">
      <>
        <nav className="bg-heroShort    h-28  bg-no-repeat bg-cover">
          <Navbar></Navbar>
        </nav>

        <main className=" bg-slate-100 grow  ">{children}</main>
      </>
    </div>
  );
};
export default DefaultLayout;
