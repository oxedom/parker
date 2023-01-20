import Sidebar from "../components/Sidebar";
import DashboardLayout from "../layouts/DashboardLayout"
import Camera from "./camera";


const Dashboard = () => {
    return ( 
    <DashboardLayout>
       <main className="w-max">
        <Camera></Camera>
       </main>
    </DashboardLayout>
  );
}
 
export default Dashboard;