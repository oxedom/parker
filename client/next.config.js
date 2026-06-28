/** @type {import('next').NextConfig} */

// The site is deployed two ways:
//   - GitHub Pages, served from a sub-path (oxedom.github.io/parker/) -> needs basePath
//   - Vercel, served from the domain root                            -> must NOT have basePath
// The Pages build runs inside GitHub Actions (GITHUB_ACTIONS=true); every other
// environment (Vercel, local dev) serves from root. Applying "/parker"
// unconditionally makes all assets/routes 404 on Vercel, leaving raw HTML.
const isGithubPages = process.env.GITHUB_ACTIONS === "true";
const basePath = isGithubPages ? "/parker" : "";

module.exports = {
  basePath,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  trailingSlash: true,
  eslint: { ignoreDuringBuilds: true },
  images: {
    unoptimized: true,
  },
};
