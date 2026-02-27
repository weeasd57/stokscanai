// Supabase Edge Function — Telegram Bot API Relay
// Proxies requests to api.telegram.org
//
// The function receives paths like:
//   /functions/v1/FUNCTION_NAME/bot<TOKEN>/sendMessage
// We extract /bot<TOKEN>/sendMessage and forward to api.telegram.org

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

serve(async (req: Request) => {
    const url = new URL(req.url);
    const path = url.pathname;

    // Health check - if no /bot in path
    if (!path.includes("/bot")) {
        return new Response(JSON.stringify({ ok: true, relay: "telegram" }), {
            headers: { "Content-Type": "application/json" },
        });
    }

    // Extract everything from /bot onwards
    // e.g. /functions/v1/dynamic-handler/bot123:ABC/sendMessage → /bot123:ABC/sendMessage
    const botIndex = path.indexOf("/bot");
    const apiPath = path.substring(botIndex);
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
