// Force Node.js runtime to avoid Edge Runtime __dirname incompatibility
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withTimeout(ms: number) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { controller, id };
}

export async function GET() {
  try {
    const base =
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      process.env.PYTHON_BACKEND_URL ||
      "http://127.0.0.1:8000";

    const { controller, id } = withTimeout(15_000);
    const res = await fetch(`${base.replace(/\/$/, "")}/admin/models/list`, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    }).finally(() => clearTimeout(id));

    if (!res.ok) {
      return Response.json(
        { models: [] },
        { status: 200 }
      );
    }

    const data = await res.json();

    // Fetch additional info for each model
    const modelsWithInfo = await Promise.all(
      (data.models || []).map(async (model: any) => {
        try {
          const { controller, id } = withTimeout(10_000);
          try {
            const infoRes = await fetch(
              `${base.replace(/\/$/, "")}/admin/models/${encodeURIComponent(model.name)}/info`,
              { method: "GET", cache: "no-store", signal: controller.signal }
            );
            if (infoRes.ok) {
              const info = await infoRes.json();
              return { ...model, ...info };
            }
          } finally {
            clearTimeout(id);
          }
        } catch (e) {
          console.error(`Error fetching info for ${model.name}:`, e);
        }
        return model;
      })
    );

    return Response.json({ models: modelsWithInfo });
  } catch (error) {
    console.error("Error fetching models:", error);
    return Response.json(
      { models: [] },
      { status: 200 }
    );
  }
}
