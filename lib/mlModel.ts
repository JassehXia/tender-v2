import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';

tf.setBackend('cpu');

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

/* ----------------------------- Tag Cleaning ----------------------------- */

function cleanTag(tag: unknown): string | null {
  if (typeof tag !== 'string') return null;

  const cleaned = tag
    .replace(/[\[\]"']/g, '')
    .trim()
    .toLowerCase();

  if (!cleaned || STOP_TAGS.has(cleaned)) return null;

  return cleaned;
}

function cleanTagList(tags: unknown[]): string[] {
  return tags
    .map(cleanTag)
    .filter((t): t is string => t !== null);
}

/* --------------------------- Feature Encoding ---------------------------- */

function oneHotEncode(
  foods: FoodItem[]
): { features: number[][]; tagCount: number } {
  const tagSet = new Set<string>();

  foods.forEach(f =>
    cleanTagList(f.tags).forEach(t => tagSet.add(t))
  );

  const tags = Array.from(tagSet);
  const tagIndex = new Map(tags.map((t, i) => [t, i]));

  const features = foods.map(food => {
    const vec = new Array(tags.length).fill(0);
    cleanTagList(food.tags).forEach(tag => {
      const idx = tagIndex.get(tag);
      if (idx !== undefined) vec[idx] = 1;
    });
    return vec;
  });

  return { features, tagCount: tags.length };
}

/* -------------------------- Main Recommendation -------------------------- */

export async function getRecommendations(
  foods: FoodItem[],
  interactions: Interaction[]
): Promise<Prediction[]> {
  if (!foods.length) return [];

  /* ---------- Cold start: no usable interactions ---------- */
  const labelsMap = new Map<string, number>();
  interactions.forEach(i => {
    if (i.action === 'LIKE') labelsMap.set(i.foodId, 1);
    if (i.action === 'DISLIKE') labelsMap.set(i.foodId, 0);
  });

  const hasSignal = labelsMap.size >= 2;

  /* ---------- Feature encoding ---------- */
  const cleanedFoods = foods.map(f => ({
    ...f,
    tags: cleanTagList(f.tags),
  }));

  const { features, tagCount } = oneHotEncode(cleanedFoods);

  /* ---------- Hard fallback: no tags ---------- */
  if (tagCount === 0 || !hasSignal) {
    return cleanedFoods.map(food => ({
      id: food.id,
      name: food.name,
      tags: food.tags,
      imageUrl: food.imageUrl,
      score: 0.5,
    }));
  }

  /* ---------- Labels ---------- */
  const labels = cleanedFoods.map(f => labelsMap.get(f.id) ?? 0);

  let X: tf.Tensor | null = null;
  let y: tf.Tensor | null = null;
  let model: tf.Sequential | null = null;

  try {
    X = tf.tensor2d(features);
    y = tf.tensor2d(labels, [labels.length, 1]);

    model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [tagCount],
        units: 8,
        activation: 'relu',
      })
    );
    model.add(
      tf.layers.dense({ units: 1, activation: 'sigmoid' })
    );

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'binaryCrossentropy',
    });

    await model.fit(X, y, {
      epochs: 30,
      shuffle: true,
      verbose: 0,
    });

    const preds = model.predict(X) as tf.Tensor;
    const scores = (await preds.array()) as number[][];

    preds.dispose();

    return cleanedFoods
      .map((food, i) => ({
        id: food.id,
        name: food.name,
        tags: food.tags,
        imageUrl: food.imageUrl,
        score: scores[i]?.[0] ?? 0.5,
      }))
      .sort((a, b) => b.score - a.score);
  } finally {
    X?.dispose();
    y?.dispose();
    model?.dispose();
  }
}
