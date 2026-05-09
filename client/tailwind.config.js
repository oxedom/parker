/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js,ts,jsx,tsx}"],
  theme: {
    extend: {
      backgroundImage: {
        hero: "linear-gradient(0deg,#3D85C6,rgba(31,6,85,.1)), url(/bg1.svg), linear-gradient(180deg,#063855,#3D85C6)",
        heroShort:
          "linear-gradient(0deg,#0b689e,rgba(31,6,85,.1)), url(/bg1.svg), linear-gradient(180deg,#063855,#0b689e)",
        filler:
          "linear-gradient(0deg,#3D85C6,rgba(31,6,85,.1)), linear-gradient(180deg,#063855,#3D85C6)",
        orangeFade:
          "linear-gradient(90deg, rgba(235,122,32,1) 66%, rgba(232,108,34,1) 74%, rgba(228,93,37,1) 100%)",
        orangeFadeSides:
          "linear-gradient(0deg, rgba(235,122,32,1) 44%, rgba(232,108,34,1) 84%, rgba(228,93,37,1) 100%)",
      },
      boxShadow: {
        neo: "0px 0px 0 black, 5px 4px 0 black",
        "neo-sm": "0px 0px 0 black, 3px 2px 0 black",
        "neo-xl": "0px 0px 0 black, 15px 13px 0 black",
        "neo-hover": "0px 0px 0 green, 7px 6px 0 black",
      },
      animation: {
        fade: "fadeIn 0.4s ease-in-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
