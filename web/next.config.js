/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
    return [
      {
        source: '/api/scan/ai',
        destination: '/api/scan/ai',
      },
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/:path*`,
      },
      {
        source: '/docs',
        destination: `${BACKEND_URL}/docs`,
      },
      {
        source: '/openapi.json',
        destination: `${BACKEND_URL}/openapi.json`,
      },
    ];
  },
};

module.exports = nextConfig;
