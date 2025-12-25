import prisma from "@/lib/prisma";
import { auth, currentUser} from "@clerk/nextjs/server"

export async function getOrCreateUser(){
    const {userId} = await auth();

    if(!userId){
        return;
    }

    const existingUser = await prisma.user.findUnique({
        where: {
            clerkId: userId
        }
    })
    if(existingUser){
        return existingUser;
    }

    const clerkUser = await currentUser();
    const email = clerkUser?.emailAddresses[0]?.emailAddress;

    if(!email){
        throw new Error('User email not found');
    }

    return await prisma.user.create({
        data:{
            clerkId: userId,
            email,
            name: clerkUser.firstName ?? ""
        }
    })
}