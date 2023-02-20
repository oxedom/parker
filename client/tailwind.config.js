/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./layouts/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./components/*.{js,ts,jsx,tsx}",
  ],

  theme: {
    extend: {
      boxShadow: 
      {
        'neo' : "0px 0px 0 black, 5px 4px 0 black",
        'neo-hover': "0px 0px 0 green, 7px 6px 0 black ",
      }


    },
  },
  plugins: [],
};
