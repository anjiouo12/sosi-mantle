export async function checkWord(word: string) {
  const response = await fetch(
    "http://127.0.0.1:8000/guess",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        word,
      }),
    }
  );


  if (!response.ok) {
    throw new Error("서버 오류");
  }


  return await response.json();
}