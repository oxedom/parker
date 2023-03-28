import "../styles/globals.css";
import { RecoilRoot } from "recoil";
import Head from "next/head";

export default function App({ Component, pageProps }) {
  return (
    <>
    <Head>
    <link rel="apple-touch-icon" sizes="180x180" href="favicon/apple-touch-icon.png"/>
<link rel="icon" type="image/png" sizes="32x32" href="favicon/favicon-32x32.png"/>
<link rel="icon" type="image/png" sizes="16x16" href="favicon/favicon-16x16.png"/>
<link rel="manifest" href="/site.webmanifest"/>
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5"/>
<meta name="msapplication-TileColor" content="#da532c"/>

    </Head>
    <RecoilRoot>
      <Component {...pageProps} />
    </RecoilRoot>
    </>

  );
}
