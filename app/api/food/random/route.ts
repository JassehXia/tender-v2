import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";

export async function GET(){
    try{
        // Count total foods
        const count = await prisma.food.count();

        if(count === 0){
            return NextResponse.json(
                {error: "No food found"},
            {status: 404})
        }

        const randomIndex = Math.floor(Math.random() * count);

        const randomFood = await prisma.food.findFirst({
            skip: randomIndex,
        })
        console.log(randomFood);
        return NextResponse.json(randomFood);

    } catch (error){
        console.log(error);
        return NextResponse.json(
            {error: "Internal Server Error"},
            {status: 500}
        )
    }
}