import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withTimeout(ms: number) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { controller, id };
}

export async function GET() {
  const base = process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";
  const targetUrl = `${base.replace(/\/$/, "")}/admin/models/list`;

  const { controller, id } = withTimeout(15_000);
  try {
    const upstream = await fetch(targetUrl, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: controller.signal,
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
    const msg = e?.name === "AbortError" ? "Upstream timeout" : "Upstream request failed";
    return NextResponse.json({ detail: msg }, { status: 502 });
  } finally {
    clearTimeout(id);
  }
}
