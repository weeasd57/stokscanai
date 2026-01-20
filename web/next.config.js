/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // On Vercel, routing is handled by vercel.json routes - skip rewrites to avoid conflicts
    if (process.env.VERCEL) {
      console.log("Running on Vercel - using vercel.json routes instead of next rewrites");
      return [];
    }

    console.log("Using Python Path:", process.env.PYTHON_PATH || "Default System Python");
    const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
    console.log("Using Python Backend URL:", BACKEND_URL);
    return [
      // Note: /api/scan/ai is handled by Next.js API route, not proxied to backend
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
    ];
  },
};

module.exports = nextConfig;
