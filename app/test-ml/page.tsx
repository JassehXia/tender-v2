/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { useState } from 'react';
import { getMLPredictions, type FoodItem, type UserInteraction } from '@/lib/mlClient';

export default function TestMLPage() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testData: FoodItem[] = [
    { id: '1', name: 'Chicken Fried Rice', tags: ['chicken', 'rice', 'fried', 'savory'], imageUrl: null },
    { id: '2', name: 'Chocolate Cake', tags: ['chocolate', 'cake', 'dessert', 'sweet'], imageUrl: null },
    { id: '3', name: 'Beef Noodles', tags: ['beef', 'noodles', 'savory', 'stir-frying'], imageUrl: null },
    { id: '4', name: 'Apple Pie', tags: ['apple', 'dessert', 'sweet', 'fruit'], imageUrl: null },
    { id: '5', name: 'Chicken Stir Fry', tags: ['chicken', 'vegetables', 'stir-frying', 'savory'], imageUrl: null },
    { id: '6', name: 'Ice Cream', tags: ['dessert', 'sweet', 'milk'], imageUrl: null },
  ];

  const testInteractions: UserInteraction[] = [
    { foodId: '1', action: 'LIKE' },
    { foodId: '2', action: 'DISLIKE' },
    { foodId: '3', action: 'LIKE' },
    { foodId: '4', action: 'DISLIKE' },
    { foodId: '5', action: 'LIKE' },
  ];

  const runTest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const predictions = await getMLPredictions(testData, testInteractions);
      setResult(predictions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Test error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Test ML Prediction API</h1>

        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Test Data</h2>
          
          <div className="mb-4">
            <h3 className="font-medium mb-2">Foods:</h3>
            <ul className="list-disc list-inside text-sm text-gray-700">
              {testData.map(food => (
                <li key={food.id}>{food.name} - {food.tags.join(', ')}</li>
              ))}
            </ul>
          </div>

          <div className="mb-4">
            <h3 className="font-medium mb-2">User Interactions:</h3>
            <ul className="list-disc list-inside text-sm text-gray-700">
              {testInteractions.map((interaction, idx) => {
                const food = testData.find(f => f.id === interaction.foodId);
                return (
                  <li key={idx}>
                    {interaction.action}: {food?.name}
                  </li>
                );
              })}
            </ul>
          </div>

          <button
            onClick={runTest}
            disabled={loading}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
          >
            {loading ? 'Running Test...' : 'Run Prediction Test'}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <h3 className="font-semibold text-red-800 mb-2">Error:</h3>
            <p className="text-red-600">{error}</p>
          </div>
        )}

        {result && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Predictions (sorted by score)</h2>
            <div className="space-y-3">
              {result.map((pred: any, idx: number) => (
                <div 
                  key={pred.id}
                  className={`p-4 rounded-lg border-2 ${
                    pred.score > 0.5 
                      ? 'bg-green-50 border-green-200' 
                      : 'bg-red-50 border-red-200'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-semibold">
                        #{idx + 1} - {pred.name}
                      </h3>
                      <p className="text-sm text-gray-600 mt-1">
                        Tags: {pred.tags.join(', ')}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">
                        {(pred.score * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {pred.score > 0.5 ? '👍 Likely to like' : '👎 Likely to dislike'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold mb-2">Analysis:</h3>
              <p className="text-sm text-gray-700">
                The model should score savory foods (chicken, beef, noodles) higher 
                and sweet/dessert foods lower based on the like/dislike pattern.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}