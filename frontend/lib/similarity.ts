export async function calculateSimilarity(
  word1: string,
  word2: string
): Promise<number> {

  const response = await fetch(
    `http://127.0.0.1:8000/similarity?input=${encodeURIComponent(word1)}&answer=${encodeURIComponent(word2)}`
  );

  if (!response.ok) {
    throw new Error("AI 서버 호출 실패");
  }

  const data = await response.json();

  return data.score;
}