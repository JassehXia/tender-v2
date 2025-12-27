'use client';

import { useState, useEffect } from 'react';
import { getPersonalizedRecommendations, recordInteraction } from '@/app/actions/recommendations';

interface Recommendation {
  id: string;
  name: string;
  tags: string[];
  imageUrl?: string | null;
  score: number;
}

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [interactionCount, setInteractionCount] = useState(0);

  useEffect(() => {
    loadRecommendations();
  }, []);

  async function loadRecommendations() {
    setLoading(true);
    setError(null);
    
    try {
      const result = await getPersonalizedRecommendations(20);
      
      if (result.success) {
        setRecommendations(result.recommendations);
        setInteractionCount(result.interactionCount || 0);
      } else {
        setError(result.message || 'Failed to load recommendations');
        setInteractionCount(result.currentCount || 0);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  async function handleLike(foodId: string) {
    try {
      await recordInteraction(foodId, 'LIKE');
      await loadRecommendations(); // Refresh with new interaction
    } catch (err) {
      console.error('Failed to record like:', err);
    }
  }

  async function handleDislike(foodId: string) {
    try {
      await recordInteraction(foodId, 'DISLIKE');
      await loadRecommendations(); // Refresh with new interaction
    } catch (err) {
      console.error('Failed to record dislike:', err);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading recommendations...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-2xl mx-auto">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-2">Not Enough Data Yet</h2>
            <p className="text-gray-700 mb-4">{error}</p>
            <p className="text-sm text-gray-600">
              You have {interactionCount} interaction(s). Need at least 5 to generate recommendations.
            </p>
            <a 
              href="/foods" 
              className="mt-4 inline-block bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Browse Foods & Start Rating
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Your Personalized Recommendations</h1>
          <p className="text-gray-600">
            Based on {interactionCount} of your ratings
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {recommendations.map((food) => (
            <div 
              key={food.id} 
              className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition"
            >
              {food.imageUrl && (
                <img 
                  src={food.imageUrl} 
                  alt={food.name}
                  className="w-full h-48 object-cover"
                />
              )}
              
              <div className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-semibold">{food.name}</h3>
                  <div className="text-right">
                    <div className="text-sm font-bold text-green-600">
                      {(food.score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">match</div>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-1 mb-4">
                  {food.tags.slice(0, 5).map((tag, idx) => (
                    <span 
                      key={idx}
                      className="text-xs bg-gray-100 px-2 py-1 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                  {food.tags.length > 5 && (
                    <span className="text-xs text-gray-500 px-2 py-1">
                      +{food.tags.length - 5} more
                    </span>
                  )}
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleLike(food.id)}
                    className="flex-1 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition"
                  >
                    👍 Like
                  </button>
                  <button
                    onClick={() => handleDislike(food.id)}
                    className="flex-1 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition"
                  >
                    👎 Dislike
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {recommendations.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            No recommendations available yet.
          </div>
        )}
      </div>
    </div>
  );
}