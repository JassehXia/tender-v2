import { PrismaClient, Prisma } from "@/app/generated/prisma/client";
import { PrismaPg } from '@prisma/adapter-pg';
import 'dotenv/config';
import fs from 'fs';

// Use your working adapter setup
const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL,
});

const prisma = new PrismaClient({ adapter });

// Load dataset JSON
const rawData = fs.readFileSync('prisma/food_dataset.json', 'utf-8');
const foodData: {
  dish_name?: string;
  food_type?: string;
  ingredients?: string;
  cooking_method?: string;
  image_url?: string;
}[] = JSON.parse(rawData);

// Map HF dataset to Prisma Food model
const mappedFoodData: Prisma.FoodCreateInput[] = foodData.map(item => {
  const tags: string[] = [];

  if (item.food_type) tags.push(item.food_type);
  if (item.ingredients) {
    // Split ingredients by comma and trim
    tags.push(...item.ingredients.split(',').map(i => i.trim()));
  }
  if (item.cooking_method) tags.push(item.cooking_method);

  return {
    name: item.dish_name || "Unknown",
    tags,
    imageUrl: item.image_url || null,
    // userId: null // optionally assign a user
  };
});

export async function main() {
  for (const f of mappedFoodData) {
    await prisma.food.create({ data: f });
  }
  console.log(`Seeded ${mappedFoodData.length} food items`);
}

main()
  .catch(e => console.error(e))
  .finally(async () => {
    await prisma.$disconnect();
  });
