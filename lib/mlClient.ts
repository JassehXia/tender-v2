// lib/mlClient.ts
export interface FoodItem {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string | null;
}

export interface UserInteraction {
  foodId: string;
  action: 'LIKE' | 'DISLIKE' | 'SAVE' | 'SKIP';
}

export interface Prediction {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string | null;
  score: number;
}

export interface PredictionResponse {
  success: boolean;
  predictions: Prediction[];
  total: number;
  error?: string;
}

export async function getMLPredictions(
  foods: FoodItem[],
  interactions: UserInteraction[]
): Promise<Prediction[]> {
  const response = await fetch('/api/food/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      foods,
      interactions,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error.error || 'Prediction failed');
  }

  const data: PredictionResponse = await response.json();
  return data.predictions;
}

export async function getRecommendations(
  foods: FoodItem[],
  interactions: UserInteraction[],
  limit: number = 10
): Promise<Prediction[]> {
  const predictions = await getMLPredictions(foods, interactions);
  return predictions.slice(0, limit);
}