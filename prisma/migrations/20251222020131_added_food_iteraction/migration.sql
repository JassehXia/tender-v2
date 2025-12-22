-- CreateEnum
CREATE TYPE "InteractionType" AS ENUM ('LIKE', 'DISLIKE', 'SAVE', 'SKIP');

-- CreateTable
CREATE TABLE "FoodInteraction" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "foodId" TEXT NOT NULL,
    "action" "InteractionType" NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "FoodInteraction_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "FoodInteraction_userId_idx" ON "FoodInteraction"("userId");

-- CreateIndex
CREATE INDEX "FoodInteraction_foodId_idx" ON "FoodInteraction"("foodId");

-- AddForeignKey
ALTER TABLE "FoodInteraction" ADD CONSTRAINT "FoodInteraction_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "FoodInteraction" ADD CONSTRAINT "FoodInteraction_foodId_fkey" FOREIGN KEY ("foodId") REFERENCES "Food"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
