'use server';

import prisma from '@/lib/prisma'
import { auth } from '@clerk/nextjs/server'

export async function getPersonalizedRecommendations(limit: number = 10) {
  const { userId } = await auth();
  
  if (!userId) {
    throw new Error('Unauthorized');
  }

  // Get user from database
  const user = await prisma.user.findUnique({
    where: { clerkId: userId },
  });

  if (!user) {
    throw new Error('User not found');
  }

  // Fetch user's interactions
  const interactions = await prisma.foodInteraction.findMany({
    where: { userId: user.id },
    select: {
      foodId: true,
      action: true,
    },
  });

  // Need at least 5 interactions
  if (interactions.length < 5) {
    return {
      success: false,
      message: 'Need at least 5 food interactions to get recommendations',
      currentCount: interactions.length,
      recommendations: [],
      interactionCount: interactions.length,
    };
  }

  // Fetch all foods
  const foods = await prisma.food.findMany({
    select: {
      id: true,
      name: true,
      tags: true,
      imageUrl: true,
    },
  });

  try {
    // Call the API route (or import getRecommendations directly)
    const baseUrl = process.env.NEXT_PUBLIC_URL || 'http://localhost:3000';
    const response = await fetch(`${baseUrl}/api/food/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ foods, interactions }),
    });

    if (!response.ok) {
      throw new Error('Failed to get predictions');
    }

    const data = await response.json();
    const predictions = data.predictions;
    
    // Return top N
    return {
      success: true,
      recommendations: predictions.slice(0, limit),
      total: predictions.length,
      interactionCount: interactions.length,
    };
  } catch (error) {
    console.error('ML Prediction error:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Failed to get recommendations',
      recommendations: [],
      interactionCount: interactions.length,
    };
  }
}

export async function recordInteraction(
  foodId: string,
  action: 'LIKE' | 'DISLIKE' | 'SAVE' | 'SKIP'
) {
  const { userId } = await auth();
  
  if (!userId) {
    throw new Error('Unauthorized');
  }

  const user = await prisma.user.findUnique({
    where: { clerkId: userId },
  });

  if (!user) {
    throw new Error('User not found');
  }

  // Check if interaction already exists
  const existingInteraction = await prisma.foodInteraction.findFirst({
    where: {
      userId: user.id,
      foodId: foodId,
    },
  });

  if (existingInteraction) {
    // Update existing interaction
    await prisma.foodInteraction.update({
      where: { id: existingInteraction.id },
      data: { action },
    });
  } else {
    // Create new interaction
    await prisma.foodInteraction.create({
      data: {
        userId: user.id,
        foodId,
        action,
      },
    });
  }

  return { success: true };
}

export async function getUserInteractionCount() {
  const { userId } = await auth();
  
  if (!userId) {
    return 0;
  }

  const user = await prisma.user.findUnique({
    where: { clerkId: userId },
  });

  if (!user) {
    return 0;
  }

  const count = await prisma.foodInteraction.count({
    where: { userId: user.id },
  });

  return count;
}