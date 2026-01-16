export async function GET() {
  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/admin/models/list`, {
      method: "GET",
    });

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
          const infoRes = await fetch(
            `${process.env.NEXT_PUBLIC_API_BASE_URL}/admin/models/${encodeURIComponent(model.name)}/info`,
            { method: "GET" }
          );
          if (infoRes.ok) {
            const info = await infoRes.json();
            return { ...model, ...info };
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
