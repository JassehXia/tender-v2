import prisma from "@/lib/prisma";
import FoodTinderSection from "./components/FoodTinderSection";

export default async function Home() {
  const users = await prisma.user.findMany();
  const currentUserId = users.length > 0 ? users[0].id : "";

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center -mt-16">
      <h1 className="text-4xl font-bold mb-8 font-[family-name:var(--font-geist-sans)] text-[#333333]">
        Superblog
      </h1>

      {/* Client-side Tinder card */}
      <FoodTinderSection userId={currentUserId} />

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
