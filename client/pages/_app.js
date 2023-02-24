import "../styles/globals.css";
import { RecoilRoot } from "recoil";
import DashboardLayout from "../layouts/DashboardLayout";

export default function App({ Component, pageProps }) {
  return (
    <RecoilRoot>
      <Component {...pageProps} />
    </RecoilRoot>
  );
}
