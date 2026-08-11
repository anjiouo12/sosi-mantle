import { NextResponse } from "next/server";
import similarityData from "@/data/similarity.json";

export async function POST(req: Request) {
  const { word, answer }: {
    word: string;
    answer: string;
  } = await req.json();


  const answerData =
  (similarityData as Record<string, Record<string, number>>)[answer];


  if (!answerData) {
    return NextResponse.json({
      score: 0,
    });
  }


  const score = answerData[word] ?? 0;

const rank =
  Object.values(answerData)
    .filter((value) => value > score)
    .length + 1;

return NextResponse.json({
  score,
  rank,
});
}