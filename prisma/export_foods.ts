import { PrismaClient } from "@/app/generated/prisma/client";
import { PrismaPg } from '@prisma/adapter-pg';
import 'dotenv/config';
import fs from 'fs';

// Prisma adapter setup
const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL,
});

const prisma = new PrismaClient({ adapter });

async function main() {
  // Pull first 10000 foods
const foods = await prisma.food.findMany({
  take: 10000,
  where: {
    tags: {
      hasSome: ["Homemade food", "Restaurant food"]
    }
  },
  select: {
    name: true,
    tags: true,
    imageUrl: true
  }
});
  // Clean tags if nested
  const cleaned = foods.map(f => ({
    name: f.name,
    tags: f.tags.flatMap(tag => {
      if (tag.startsWith('[') && tag.endsWith(']')) {
        try { return JSON.parse(tag.replace(/\\'/g, "'")); }
        catch { return [tag]; }
      }
      return [tag];
    }),
    imageUrl: f.imageUrl
  }));

  // Export to JSON
  fs.writeFileSync('food_sample.json', JSON.stringify(cleaned, null, 2));
  console.log(`Exported ${cleaned.length} foods to food_sample.json`);
}

main()
  .catch(e => console.error(e))
  .finally(async () => await prisma.$disconnect());
