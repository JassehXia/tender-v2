import prisma from '@/lib/prisma'
import dynamic from "next/dynamic";

// Dynamically import the client-side FoodTinderCard
const FoodTinderCard = dynamic(() => import("./components/food-card"), {
  ssr: false,
});

export default async function Home() {
  const users = await prisma.user.findMany();

  // For demonstration, pick the first user as "current user"
  const currentUserId = users.length > 0 ? users[0].id : "";

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center -mt-16">
      <h1 className="text-4xl font-bold mb-8 font-[family-name:var(--font-geist-sans)] text-[#333333]">
        Superblog
      </h1>

      <FoodTinderCard userId={currentUserId} />

      <ol className="list-decimal list-inside font-[family-name:var(--font-geist-sans)] mt-8">
        {users.map((user) => (
          <li key={user.id} className="mb-2">
            {user.name}
          </li>
        ))}
      </ol>
    </div>
  );
}
