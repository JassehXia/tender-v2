import { getOrCreateUser } from "@/lib/getOrCreateUser";
import FoodTinderSection from "./components/FoodTinderSection";
import { SignedIn, SignedOut } from '@clerk/nextjs'

export default async function Home() {
  // Ensure the Prisma user exists for the signed-in Clerk user
  await getOrCreateUser();


  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center -mt-16">
      <SignedIn>
      <FoodTinderSection />
      </SignedIn>
      <SignedOut>
      <div className="text-2xl font-bold text-gray-700">Sign in to start using Tender</div>
      </SignedOut>
    </div>
  );
}
