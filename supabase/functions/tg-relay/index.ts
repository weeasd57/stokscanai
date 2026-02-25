// Supabase Edge Function — Telegram Bot API Relay
// Proxies requests from HF Spaces to api.telegram.org
//
// Deploy:
//   supabase functions deploy tg-relay
// Or via Supabase Dashboard → Edge Functions → Create → paste this

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

serve(async (req: Request) => {
    const url = new URL(req.url);

    // Health check
    if (url.pathname === "/" || url.pathname === "/tg-relay") {
        return new Response(JSON.stringify({ ok: true, relay: "telegram" }), {
            headers: { "Content-Type": "application/json" },
        });
    }

    // Extract the path after /tg-relay/
    // e.g. /tg-relay/bot<TOKEN>/sendMessage → /bot<TOKEN>/sendMessage
    let apiPath = url.pathname;
    if (apiPath.startsWith("/tg-relay")) {
        apiPath = apiPath.replace("/tg-relay", "");
    }

    const telegramUrl = `https://api.telegram.org${apiPath}${url.search}`;

    try {
        const init: RequestInit = {
            method: req.method,
            headers: { "Content-Type": "application/json" },
        };

        if (req.method !== "GET" && req.method !== "HEAD") {
            init.body = await req.text();
        }

        const resp = await fetch(telegramUrl, init);
        const body = await resp.text();

        return new Response(body, {
            status: resp.status,
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
        });
    } catch (err) {
        return new Response(
            JSON.stringify({ ok: false, error: err.message }),
            { status: 502, headers: { "Content-Type": "application/json" } }
        );
    }
});
