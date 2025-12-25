import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import prisma from "@/lib/prisma";

type InteractionType = "LIKE" | "DISLIKE" | "SAVE" | "SKIP";

export async function POST(req: Request) {
  try {
    const { userId } = await auth();

    if (!userId) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const { foodId, action } = await req.json();

    if (!foodId || !action) {
      return NextResponse.json(
        { error: "foodId and action are required" },
        { status: 400 }
      );
    }

    const validActions: InteractionType[] = ["LIKE", "DISLIKE", "SAVE", "SKIP"];
    if (!validActions.includes(action)) {
      return NextResponse.json(
        { error: "Invalid action" },
        { status: 400 }
      );
    }

    // SKIP doesn't persist
    if (action === "SKIP") {
      return NextResponse.json({ success: true });
    }

    const existing = await prisma.foodInteraction.findFirst({
      where: { userId, foodId },
    });

    if (existing) {
      if (existing.action === action) {
        return NextResponse.json(
          { message: `Already ${action.toLowerCase()}` },
          { status: 200 }
        );
      }

      await prisma.foodInteraction.update({
        where: { id: existing.id },
        data: { action },
      });

      return NextResponse.json({
        success: true,
        message: "Interaction updated",
      });
    }

    await prisma.foodInteraction.create({
      data: {
        userId,
        foodId,
        action,
      },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
