// /** @type {import('next').NextConfig} */
// const nextConfig = {
//   // reactStrictMode: true,
//   webpack(config) {
//     config.module.rules.push({
//       test: /\.svg$/i,
//       issuer: /\.[jt]sx?$/,
//       use: ["@svgr/webpack"],
//     });

//     return config;
//   },
// };

// module.exports = nextConfig;
module.exports = {
  eslint: { ignoreDuringBuilds: true },
  images: {
    unoptimized: true,
  },
  /* config options here */
};
