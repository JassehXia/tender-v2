// components/FoodTinderCard.tsx
"use client";

import { useEffect, useState } from "react";
import Image from "next/image";

type Food = {
  id: string;
  name: string;
  description?: string;
  imageUrl?: string;
};

type Interaction = "LIKE" | "DISLIKE" | "SAVE" | "SKIP";

export default function FoodTinderCard({ userId }: { userId: string }) {
  const [food, setFood] = useState<Food | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchRandomFood = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/food/random");
      if (!res.ok) throw new Error("Failed to fetch food");
      const data = await res.json();
      setFood(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInteraction = async (action: Interaction) => {
    if (!food) return;

    try {
      await fetch("/api/food/interact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId,
          foodId: food.id,
          action,
        }),
      });
      fetchRandomFood();
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchRandomFood();
  }, []);

  if (loading) return <div className="text-center p-4">Loading...</div>;
  if (!food) return <div className="text-center p-4">No food available</div>;

  return (
    <div className="max-w-sm mx-auto bg-white shadow-lg rounded-lg overflow-hidden my-8">
      {food.imageUrl && (
        <div className="relative w-full h-64">
          <Image
            src={food.imageUrl}
            alt={food.name}
            fill
            className="object-cover rounded-t-lg"
            priority
          />
        </div>
      )}
      <div className="p-4">
        <h2 className="text-xl font-bold mb-2">{food.name}</h2>
        <p className="text-gray-600 mb-4">{food.description}</p>
        <div className="flex justify-between">
          <button
            onClick={() => handleInteraction("DISLIKE")}
            className="bg-red-500 text-white px-4 py-2 rounded"
          >
            ❌
          </button>
          <button
            onClick={() => handleInteraction("SKIP")}
            className="bg-gray-500 text-white px-4 py-2 rounded"
          >
            ⏭
          </button>
          <button
            onClick={() => handleInteraction("SAVE")}
            className="bg-blue-500 text-white px-4 py-2 rounded"
          >
            💾
          </button>
          <button
            onClick={() => handleInteraction("LIKE")}
            className="bg-green-500 text-white px-4 py-2 rounded"
          >
            ❤️
          </button>
        </div>
      </div>
    </div>
  );
}
