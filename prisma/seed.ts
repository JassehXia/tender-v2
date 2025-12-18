import { PrismaClient, Prisma } from "@/app/generated/prisma/client";
import { PrismaPg } from '@prisma/adapter-pg';
import 'dotenv/config';
import fs from 'fs';

// Prisma adapter setup
const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL,
});

const prisma = new PrismaClient({ adapter });

// Load dataset JSON
const rawData = fs.readFileSync('prisma/food_dataset.json', 'utf-8');
const foodData: {
  name: string;
  tags: string[];
  imageUrl?: string;
}[] = JSON.parse(rawData);

// Helper: clean tags by parsing stringified arrays
function cleanTags(tags: string[]): string[] {
  const cleaned: string[] = [];
  for (let tag of tags) {
    tag = tag.trim();

    // If it looks like a JSON array, try to parse it
    if (tag.startsWith('[') && tag.endsWith(']')) {
      try {
        const parsed = JSON.parse(tag.replace(/\\'/g, "'"));
        if (Array.isArray(parsed)) {
          cleaned.push(...parsed.map(t => t.trim()));
          continue;
        }
      } catch {
        // fallback if parsing fails
      }
    }

    // Push the tag normally
    cleaned.push(tag);
  }
  return cleaned;
}

// Map JSON to Prisma Food model
const mappedFoodData: Prisma.FoodCreateInput[] = foodData.map(item => ({
  name: item.name,
  tags: cleanTags(item.tags),
  imageUrl: item.imageUrl || null,
}));

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
