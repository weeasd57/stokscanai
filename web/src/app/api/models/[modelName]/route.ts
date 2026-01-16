export async function DELETE(
  request: Request,
  { params }: { params: { modelName: string } }
) {
  try {
    const { modelName } = params;

    const res = await fetch(
      `${process.env.NEXT_PUBLIC_API_BASE_URL}/admin/models/${encodeURIComponent(modelName)}`,
      {
        method: "DELETE",
      }
    );

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      return Response.json(
        { status: "error", message: errorData.detail || "Failed to delete model" },
        { status: res.status }
      );
    }

    const data = await res.json();
    return Response.json(data);
  } catch (error) {
    console.error("Error deleting model:", error);
    return Response.json(
      { status: "error", message: "Connection error" },
      { status: 500 }
    );
  }
}
