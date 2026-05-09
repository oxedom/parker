/** @type {import('next').NextConfig} */
module.exports = {
  basePath: "/parker",
  assetPrefix: "/parker/",
  trailingSlash: true,
  eslint: { ignoreDuringBuilds: true },
  images: {
    unoptimized: true,
  },
};
