import * as tf from '@tensorflow/tfjs';

const STOP_TAGS = new Set(['homemade food', 'restaurant food']);

interface FoodItem {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string | null;
}

interface Interaction {
  foodId: string;
  action: 'LIKE' | 'DISLIKE' | 'SAVE' | 'SKIP';
}

interface Prediction {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string | null;
  score: number;
}

function cleanTag(tag: string): string | null {
  if (typeof tag !== 'string') return null;
  
  const cleaned = tag
    .replace(/[\[\]"']/g, '')
    .trim()
    .toLowerCase();
  
  if (STOP_TAGS.has(cleaned) || cleaned === '') return null;
  
  return cleaned;
}

function cleanTagList(tags: string[]): string[] {
  return tags
    .map(cleanTag)
    .filter((tag): tag is string => tag !== null);
}

function oneHotEncode(foods: FoodItem[]): { encoded: number[][], allTags: string[] } {
  // Get all unique tags
  const tagSet = new Set<string>();
  foods.forEach(food => {
    cleanTagList(food.tags).forEach(tag => tagSet.add(tag));
  });
  
  const allTags = Array.from(tagSet).sort();
  const tagToIndex = new Map(allTags.map((tag, idx) => [tag, idx]));
  
  // Create one-hot encoded matrix
  const encoded = foods.map(food => {
    const vector = new Array(allTags.length).fill(0);
    cleanTagList(food.tags).forEach(tag => {
      const idx = tagToIndex.get(tag);
      if (idx !== undefined) {
        vector[idx] = 1;
      }
    });
    return vector;
  });
  
  return { encoded, allTags };
}

export async function getRecommendations(
  foods: FoodItem[],
  interactions: Interaction[]
): Promise<Prediction[]> {
  // Clean tags
  const cleanedFoods = foods.map(food => ({
    ...food,
    tags: cleanTagList(food.tags)
  }));
  
  // One-hot encode
  const { encoded, allTags } = oneHotEncode(cleanedFoods);
  
  // Create labels from interactions
  const interactionMap = new Map<string, number>();
  interactions.forEach(interaction => {
    if (interaction.action === 'LIKE') {
      interactionMap.set(interaction.foodId, 1);
    } else if (interaction.action === 'DISLIKE') {
      interactionMap.set(interaction.foodId, 0);
    }
  });
  
  const labels = cleanedFoods.map(food => 
    interactionMap.get(food.id) ?? 0
  );
  
  // Convert to tensors
  const X = tf.tensor2d(encoded);
  const y = tf.tensor2d(labels, [labels.length, 1]);
  
  // Build model
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [allTags.length], units: 8, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'sigmoid' })
    ]
  });
  
  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  // Train model
  await model.fit(X, y, {
    epochs: 50,
    verbose: 0,
    shuffle: true
  });
  
  // Make predictions
  const predictions = model.predict(X) as tf.Tensor;
  const scores = await predictions.array() as number[][];
  
  // Clean up tensors
  X.dispose();
  y.dispose();
  predictions.dispose();
  model.dispose();
  
  // Prepare results
  const results: Prediction[] = cleanedFoods.map((food, i) => ({
    id: food.id,
    name: food.name,
    tags: food.tags,
    imageUrl: food.imageUrl,
    score: scores[i][0]
  }));
  
  // Sort by score (highest first)
  results.sort((a, b) => b.score - a.score);
  
  return results;
}