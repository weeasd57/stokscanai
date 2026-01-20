import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  try {
    if (request.nextUrl.pathname.startsWith("/api")) {
      return NextResponse.next();
    }

    // Temporarily disabled for debugging Vercel 500 error
    /*
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY;

    // If env vars are missing, just proceed without blocking
    if (!url || !anonKey) {
      console.warn("Middleware: Missing Supabase env vars");
      return NextResponse.next();
    }

    let response = NextResponse.next({
      request: {
        headers: request.headers,
      },
    });

    const supabase = createServerClient(url, anonKey, {
      cookies: {
        get(name: string) {
          return request.cookies.get(name)?.value;
        },
        set(name: string, value: string, options: any) {
          try {
            response.cookies.set({ name, value, ...options });
          } catch (e) {
            // Ignore cookie errors
          }
        },
        remove(name: string, options: any) {
          try {
            response.cookies.set({ name, value: "", ...options, maxAge: 0 });
          } catch (e) {
             // Ignore cookie errors
          }
        },
      },
    });

    // Refresh session if needed
    await supabase.auth.getUser();

    return response;
    */
    return NextResponse.next();
  } catch (e) {
    // If anything fails in middleware, don't crash the app, just let the request through
    console.error("Middleware failed:", e);
    return NextResponse.next();
  }
}

export const config = {
  // Update matcher to cleanly exclude api, static, image, favicon
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
    "/((?!api|_next/static|_next/image|favicon.ico|symbols|scan).*)",
  ],
};
