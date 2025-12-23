import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";

type InteractionType = "LIKE" | "DISLIKE" | "SAVE" | "SKIP"

export async function POST(req: Request){
    try{
        const body = await req.json();
        const {userId, foodId, action} = body;

        if(!userId || !foodId || !action){
            return NextResponse.json(
                {error: "userId, foodId, and action are required"},
                {status: 400}
            )
        }
        // Validate the action
        const validActions: InteractionType[] = ["LIKE", "DISLIKE", "SAVE", "SKIP"];
        if(!validActions.includes(action as InteractionType)){
            return NextResponse.json({error: "Invalid action"},
                {status: 400}
            )
        }

        if(action === "SKIP"){
            return NextResponse.json({success: true});
        }

        // Check if interaction already exists
        const existing = await prisma.foodInteraction.findFirst({
            where: {userId, foodId},
        });
        if(existing){
            if(existing.action === action){
                return NextResponse.json({message: `Already ${action.toLowerCase()}`}, {status: 200})
            } else{
                await prisma.foodInteraction.update({
                    where:{id: existing.id},
                    data:{ action }
                });
            }
            return NextResponse.json(
                {success:true, message: "Interaction Updated"}
            )
        }
        await prisma.foodInteraction.create({
                data: {userId, foodId, action}
            })
    return NextResponse.json({ success: true });
    } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}