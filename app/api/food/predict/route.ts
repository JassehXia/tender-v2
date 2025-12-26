import { NextRequest, NextResponse } from 'next/server';
import { getRecommendations } from '@/lib/mlModel';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { foods, interactions } = body;

    // Validate input
    if (!foods || !Array.isArray(foods)) {
      return NextResponse.json(
        { error: 'Foods array is required' },
        { status: 400 }
      );
    }

    if (!interactions || !Array.isArray(interactions) || interactions.length < 5) {
      return NextResponse.json(
        { error: 'Need at least 5 interactions to train the model' },
        { status: 400 }
      );
    }

    // Get predictions
    const predictions = await getRecommendations(foods, interactions);

    return NextResponse.json({
      success: true,
      predictions,
      total: predictions.length,
    });
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { 
        success: false,
        error: error instanceof Error ? error.message : 'Prediction failed' 
      },
      { status: 500 }
    );
  }
}