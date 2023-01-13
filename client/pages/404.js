import Link from 'next/link'
import { useEffect } from 'react';
import {useRouter } from 'next/router'

const NotFound = () => {

    const router = useRouter()
    
    useEffect(() => {
        setTimeout(() => {
            router.push('/')
        }, 3000)

    },   [])
    return ( <div className='bg-green-500'>
        <h1>Oooops.... </h1>
        <h2> That page can not be found</h2>
        <p> Go back to the</p>  <Link href='/' > Home </Link>
    </div>  );
}
 
export default NotFound;