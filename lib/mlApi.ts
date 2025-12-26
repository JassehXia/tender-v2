/* eslint-disable @typescript-eslint/no-explicit-any */
// lib/mlApi.ts
const API_URL = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000';

export interface FoodItem {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string;
}

export interface UserInteraction {
  foodId: string;
  action: 'LIKE' | 'DISLIKE' | 'SAVE' | 'SKIP';
}

export interface PredictionResponse {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string;
  score: number;
}

class MLApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `Request failed: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck() {
    return this.request<{ status: string; model_loaded: boolean }>('/health');
  }

  async trainModel(foods: FoodItem[], interactions: UserInteraction[]) {
    return this.request<{ message: string; details: any }>('/train', {
      method: 'POST',
      body: JSON.stringify({ foods, interactions }),
    });
  }

  async predict(foods: FoodItem[]): Promise<PredictionResponse[]> {
    return this.request<PredictionResponse[]>('/predict', {
      method: 'POST',
      body: JSON.stringify({ foods }),
    });
  }

  async getRecommendations(foods: FoodItem[], topN: number = 10) {
    return this.request<{
      recommendations: PredictionResponse[];
      total_evaluated: number;
    }>(`/recommend?top_n=${topN}`, {
      method: 'POST',
      body: JSON.stringify({ foods }),
    });
  }

  async saveModel() {
    return this.request<{ message: string }>('/save-model', {
      method: 'POST',
    });
  }

  async loadModel() {
    return this.request<{ message: string }>('/load-model', {
      method: 'POST',
    });
  }
}

export const mlApi = new MLApiClient();

// Example usage in a React component:
/*
import { mlApi } from '@/lib/mlApi';

// In your component or server action:
async function trainUserModel(userId: string) {
  // Fetch user's food interactions from your database
  const interactions = await prisma.foodInteraction.findMany({
    where: { userId },
  });

  // Fetch all foods
  const foods = await prisma.food.findMany();

  // Train the model
  const result = await mlApi.trainModel(foods, interactions);
  return result;
}

async function getRecommendations() {
  // Fetch all foods
  const foods = await prisma.food.findMany();
  
  // Get recommendations
  const recommendations = await mlApi.getRecommendations(foods, 10);
  return recommendations;
}
*/