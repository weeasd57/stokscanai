import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
    // Check if we are in the edge environment
    console.log('Middleware hit:', request.nextUrl.pathname);
    return NextResponse.next();
}

export const config = {
    matcher: [
        /*
         * Match all request paths except for the ones starting with:
         * - api (API routes)
         * - _next/static (static files)
         * - _next/image (image optimization files)
         * - favicon.ico (favicon file)
         * - symbols (API proxy routes)
         * - scan (API proxy routes)
         */
        '/((?!api|_next/static|_next/image|favicon.ico|symbols|scan).*)',
    ],
};
