/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/scan/ai',
        destination: '/api/scan/ai',
      },
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/:path*',
      },
      {
        source: '/docs',
        destination: 'http://127.0.0.1:8000/docs',
      },
      {
        source: '/openapi.json',
        destination: 'http://127.0.0.1:8000/openapi.json',
      },
    ];
  },
};

module.exports = nextConfig;
