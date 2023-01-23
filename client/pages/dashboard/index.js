
import Camera from '../../components/camera'
import { useRouter } from 'next/router'

const Dashboard = () => {
  const router = useRouter()

  if(router.pathname == "/dashboard")
  {return <h1> Hello</h1>}

  if(router.pathname == "/dashboard/parking")
  {return <Camera/>}

  if(router.pathname == "/dashboard/fencing")
  return <Fencing/>
}

export default Dashboard