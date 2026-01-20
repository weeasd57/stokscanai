import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  const base = process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";
  const targetUrl = `${base.replace(/\/$/, "")}/admin/models/list`;

  try {
    const upstream = await fetch(targetUrl, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    const contentType = upstream.headers.get("content-type") || "application/json";
    if (upstream.body) {
      return new Response(upstream.body, {
        status: upstream.status,
        headers: { "content-type": contentType },
      });
    }

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": contentType },
    });
  } catch (e: any) {
    return NextResponse.json({ detail: "Upstream request failed" }, { status: 502 });
  }
}
