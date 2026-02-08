/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    console.log("Using Python Path:", process.env.PYTHON_PATH || "Default System Python");
    const BACKEND_URL =
      process.env.PYTHON_BACKEND_URL ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      "http://127.0.0.1:8000";
    console.log("Using Python Backend URL:", BACKEND_URL);
    return [
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
      // Proxy specific top-level routes to backend
      {
        source: '/symbols/:path*',
        destination: `${BACKEND_URL}/symbols/:path*`,
      },
      {
        source: '/scan/:path*',
        destination: `${BACKEND_URL}/scan/:path*`,
      },
      {
        source: '/predict',
        destination: `${BACKEND_URL}/predict`,
      },
      {
        source: '/models/:path*',
        destination: `${BACKEND_URL}/models/:path*`,
      },
      {
        source: '/news',
        destination: `${BACKEND_URL}/news`,
      },
      {
        source: '/price',
        destination: `${BACKEND_URL}/price`,
      },
      {
        source: '/health',
        destination: `${BACKEND_URL}/health`,
      },
      {
        source: '/positions/:path*',
        destination: `${BACKEND_URL}/positions/:path*`,
      },
      {
        source: '/bot/:path*',
        destination: `${BACKEND_URL}/bot/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
