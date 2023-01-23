import { useState, useEffect } from 'react';
import { useRouter } from 'next/router'
import security_icon from '../public/icons/security_icon.svg'
import Link from 'next/link'
const Sidebar = () => {

    const router = useRouter()

    const [visionOpen, setVisionOpen] = useState(false);
    const [docsOpen, setDocsOpen] = useState(false);
  
    return (
      <aside className="bg-blue-200 h-screen w-80 flex flex-col items-center">

        <header className='pt-10 '>
        <h2 className="text-5xl" > Parker </h2>
        </header>

        <div className='bg-green-500 justify-self-center place-self-center self-center'>

      
        <div className='flex-col flex items-center  '>
          <h3 className="text-3xl cursor-pointer" onClick={() => setVisionOpen(!visionOpen)}>
            Dashboard
            <span className={`${visionOpen ? "rotate-180" : ""} ml-2 fas fa-caret-down`} />
          </h3>
          <div
            className={`transition-all duration-500 flex flex-col ${
              visionOpen ? "max-h-64" : "max-h-0"
            } overflow-hidden`}
          >
             <Link href="/dashboard/parking">Parking Mode</Link>
             <Link href="/dashboard/fencing">Fencing Mode</Link>
             <Link href="/dashboard/crowd">Crowd Mode</Link>
          </div>
  
          <h3 className="text-3xl cursor-pointer" onClick={() => setDocsOpen(!docsOpen)}>
            Docs
            <span className={`${docsOpen ? "rotate-180" : ""} ml-2 fas fa-caret-down`} />
          </h3>
          <div
            className={`transition-all duration-500 flex flex-col ${
              docsOpen ? "max-h-64" : "max-h-0"
            } overflow-hidden`}
          >
        <Link href="/docs/setup">Getting Started</Link>
        <Link href="/docs/modes">Vision modes </Link>
     
          </div>
        </div>
        </div>
      </aside>
    );
  };
  
  export default Sidebar;