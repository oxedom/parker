import { useState, useEffect } from "react";
import { useRouter } from "next/router";
import dropDown from "../static/icons/down-arrow.png";
import Image from "next/image";

import Link from "next/link";
import { Imag } from "@tensorflow/tfjs";
const Sidebar = () => {
  const router = useRouter();

  const [visionOpen, setVisionOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);

  const handleDocsOpen = () => {
    setDocsOpen(!docsOpen);
  };
  const handleVisionOpen = () => {
    setVisionOpen(!visionOpen);
  };

  return (
    <aside className="bg-blue-200 h-screen w-80 flex flex-col items-center">
      <header className="pt-10 ">
        <h2 className="text-5xl"> Parker </h2>
      </header>

      <div className="bg-green-500 justify-self-center place-self-center self-center">
        <div className="flex-col flex items-center cursor-pointer  ">
          <div className="flex" onClick={() => handleVisionOpen()}>
            <h3 className="text-3xl ">
              Dashboard
              <span
                className={`${
                  visionOpen ? "rotate-180" : ""
                } ml-2 fas fa-caret-down`}
              />
            </h3>
            <Image src={dropDown} width={50} height={50} />
          </div>

          <section
            className={`transition-all duration-300 flex flex-col ${
              visionOpen ? "max-h-64" : "max-h-0"
            } overflow-hidden`}
          >
            <Link href="/dashboard/parking">Parking Mode</Link>
            <Link href="/dashboard/fencing">Fencing Mode</Link>
            <Link href="/dashboard/crowd">Crowd Mode</Link>
          </section>
        </div>

        <div className="flex-col flex items-center cursor-pointer  ">
          <div className="flex" onClick={() => handleDocsOpen()}>
            <h3 className="text-3xl ">
              Docs
              <span
                className={`${
                  visionOpen ? "rotate-180" : ""
                } ml-2 fas fa-caret-down`}
              />
            </h3>
            <Image src={dropDown} width={50} height={50} />
          </div>

          <section
            className={`transition-all duration-300 flex flex-col ${
              visionOpen ? "max-h-64" : "max-h-0"
            } overflow-hidden`}
          >
            <Link href="/docs/setup">Getting Started</Link>
            <Link href="/docs/modes">Vision modes </Link>
          </section>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
