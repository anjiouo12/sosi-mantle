"use client";

import { useState, useRef, useEffect } from "react";

type Guess = {
  word: string;
  score: number;
  rank: number;
};

type RankItem = {
  rank: number;
  word: string;
  score: number;
};

export default function Home() {
  const [input, setInput] = useState("");
  const [guesses, setGuesses] = useState<Guess[]>([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  // 정답 맞힘 여부 및 모달 관련 상태
  const [isGameWon, setIsGameWon] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [topRanks, setTopRanks] = useState<RankItem[]>([]);
  const [loadingRanks, setLoadingRanks] = useState(false);
  const [resetting, setResetting] = useState(false);

  // 하루 게임 플레이 제한 (최대 3판) 관련 상태
  const [completedGames, setCompletedGames] = useState<number>(0);

  // 입력창 포커스 유지를 위한 Ref
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();

    const today = new Date().toISOString().slice(0, 10);
    const savedData = JSON.parse(localStorage.getItem("game_play_limit") || "{}");

    if (savedData.date !== today) {
      setCompletedGames(0);
      localStorage.setItem("game_play_limit", JSON.stringify({ date: today, games: 0 }));
    } else {
      setCompletedGames(savedData.games || 0);
    }
  }, []);

  const getScoreBadge = (rank: number, score: number) => {
    if (rank === 1) return { emoji: "🎉", color: "bg-red-500", text: "text-red-500" };
    if (rank <= 10) return { emoji: "🔥", color: "bg-orange-500", text: "text-orange-500" };
    if (rank <= 100) return { emoji: "✨", color: "bg-amber-500", text: "text-amber-500" };
    if (rank <= 500) return { emoji: "💡", color: "bg-yellow-400", text: "text-yellow-600" };
    if (score > 300) return { emoji: "🌱", color: "bg-emerald-400", text: "text-emerald-600" };
    return { emoji: "❄️", color: "bg-blue-300", text: "text-blue-500" };
  };

  async function fetchTopRanks() {
    setLoadingRanks(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/top-ranks?limit=100`);
      if (res.ok) {
        const data = await res.json();
        setTopRanks(data.top_ranks || []);
        setShowModal(true);
      }
    } catch (e) {
      console.error(e);
      alert("전체 순위를 불러오는데 실패했습니다.");
    } finally {
      setLoadingRanks(false);
    }
  }

  // 다음 문제로 넘어가는 정답 리셋 함수
  async function handleNextGame() {
    if (completedGames >= 3) {
      alert("오늘 할당된 3번의 기회를 모두 완료하셨습니다!");
      return;
    }

    setResetting(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/reset-answer`, { method: "POST" });

      if (res.ok) {
        setGuesses([]);
        setIsGameWon(false);
        setMessage("");
        setInput("");
        setTimeout(() => inputRef.current?.focus(), 50);
      } else {
        alert("새 문제를 불러오지 못했습니다.");
      }
    } catch (e) {
      console.error(e);
      alert("서버 통신에 실패했습니다.");
    } finally {
      setResetting(false);
    }
  }

  async function handleSubmit() {
    const trimmedInput = input.trim();

    if (!trimmedInput || loading) return;

    if (completedGames >= 3) {
      setMessage("⚠️ 오늘 플레이할 수 있는 3번의 기회를 모두 사용하셨습니다. 내일 다시 도전해 주세요!");
      setInput("");
      return;
    }

    const koreanRegex = /^[가-힣]+$/;
    if (!koreanRegex.test(trimmedInput)) {
      setMessage("⚠️ 완성된 한글 단어만 입력해 주세요.");
      return;
    }

    // 💡 1. 이미 입력한 단어인지 확인
    const existingGuess = guesses.find((g) => g.word === trimmedInput);
    if (existingGuess) {
      const rankText = existingGuess.rank === 9999 ? "10,000위 밖" : `${existingGuess.rank}위`;
      
      // 이미 입력한 단어도 상단 메시지로 알림 (정렬 순서는 그대로 유두)
      setMessage(`ℹ️ 이미 입력했던 단어입니다. (${trimmedInput} - 점수: ${existingGuess.score}점 / 순위: ${rankText})`);
      setInput("");
      setTimeout(() => inputRef.current?.focus(), 50);
      return;
    }

    setMessage("");
    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/guess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ guess: trimmedInput }),
      });

      if (!response.ok) throw new Error("서버 통신 오류");

      const result = await response.json();

      if (!result.exists) {
        setMessage(`❌ ${result.message || "사전에 등록되지 않은 단어입니다."}`);
        setInput("");
        return;
      }

      const newGuess: Guess = {
        word: trimmedInput,
        score: result.score,
        rank: result.rank,
      };

      // 💡 2. 신규 제출 단어를 추가한 후 높은 점수순(내림차순) 정렬
      setGuesses((prev) => [...prev, newGuess].sort((a, b) => b.score - a.score));
      setInput("");

      const rankText = result.rank === 9999 ? "10,000위 밖" : `${result.rank}위`;

      // 💡 3. 정답 여부에 따른 상단 메시지 표기 (신규 단어도 점수 및 랭크 표기)
      if (result.answer) {
        setMessage(`🎉 정답입니다! (${trimmedInput} - 점수: ${result.score}점 / 순위: ${rankText})`);
        setIsGameWon(true);

        const today = new Date().toISOString().slice(0, 10);
        const newGameCount = completedGames + 1;
        setCompletedGames(newGameCount);
        localStorage.setItem("game_play_limit", JSON.stringify({ date: today, games: newGameCount }));
      } else {
        setMessage(`💡 [${trimmedInput}] 점수: ${result.score}점 / 순위: ${rankText}`);
      }
    } catch (error) {
      console.error(error);
      setMessage("⚠️ 서버 통신 실패 (백엔드 서버를 확인하세요)");
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-50">
      <div className="w-full max-w-lg bg-white p-8 rounded-2xl shadow-xl flex flex-col items-center">
        <h1 className="text-3xl font-extrabold mb-2 text-center text-gray-800 flex items-center gap-2">
          소시맨틀 🔥
        </h1>
        <p className="text-sm text-gray-500 mb-1 text-center">
          단어를 입력해서 정답과 얼마나 가까운지 맞춰보세요.
        </p>

        <p className="text-xs font-medium text-amber-600 mb-6 text-center">
          💡 하루에 최대 3개의 문제(게임)까지 도전 가능합니다. (오늘 완료한 문제: {completedGames} / 3)
        </p>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
          className="flex gap-2 w-full mb-3"
        >
          <input
            ref={inputRef}
            type="text"
            className="flex-1 border border-gray-300 px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-black text-black disabled:bg-gray-100"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading || completedGames >= 3 || isGameWon}
            placeholder={
              completedGames >= 3
                ? "오늘 도전 기회를 모두 소모했습니다."
                : isGameWon
                ? "정답을 맞히셨습니다!"
                : "단어 입력..."
            }
            autoFocus
          />

          <button
            type="submit"
            disabled={loading || completedGames >= 3 || isGameWon}
            className="bg-black text-white px-6 py-3 rounded-xl font-bold hover:bg-gray-800 disabled:bg-gray-400 transition cursor-pointer disabled:cursor-not-allowed"
          >
            {loading ? "확인중" : "입력"}
          </button>
        </form>

        {message && (
          <div className="mt-1 flex flex-col items-center gap-2 w-full">
            <p className="text-sm font-semibold text-center text-indigo-600 bg-indigo-50 px-4 py-2 rounded-lg w-full border border-indigo-100">
              {message}
            </p>
            {isGameWon && (
              <div className="flex gap-2 mt-1">
                <button
                  type="button"
                  onClick={fetchTopRanks}
                  disabled={loadingRanks}
                  className="bg-amber-500 text-white text-xs px-4 py-2.5 rounded-lg font-bold hover:bg-amber-600 transition shadow cursor-pointer"
                >
                  {loadingRanks ? "불러오는 중..." : "🏆 TOP 100 보기"}
                </button>

                {completedGames < 3 && (
                  <button
                    type="button"
                    onClick={handleNextGame}
                    disabled={resetting}
                    className="bg-emerald-600 text-white text-xs px-4 py-2.5 rounded-lg font-bold hover:bg-emerald-700 transition shadow cursor-pointer"
                  >
                    {resetting ? "새 문제 준비중..." : "➡️ 다음 문제 도전하기"}
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        <div className="mt-6 w-full space-y-3">
          {guesses.map((g) => {
            const { emoji, color, text } = getScoreBadge(g.rank, g.score);
            const progressPercent = Math.min(Math.max((g.score / 1000) * 100, 5), 100);

            return (
              <div
                key={g.word}
                className="bg-gray-50 p-3 rounded-xl border border-gray-100 flex flex-col gap-1 shadow-sm transition-all"
              >
                <div className="flex justify-between items-center text-sm font-bold">
                  <span className="flex items-center gap-1.5 text-gray-800">
                    <span>{emoji}</span>
                    <span>{g.word}</span>
                    <span className="text-xs text-gray-400 font-normal">
                      ({g.rank === 9999 ? "10,000위 밖" : `${g.rank}위`})
                    </span>
                  </span>
                  <span className={`${text}`}>{g.score}점</span>
                </div>

                <div className="w-full bg-gray-200 h-2.5 rounded-full overflow-hidden mt-1">
                  <div
                    className={`h-full ${color} transition-all duration-500 rounded-full`}
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl w-full max-w-md p-6 max-h-[80vh] flex flex-col shadow-2xl">
            <div className="flex justify-between items-center border-b pb-3 mb-4">
              <h2 className="text-lg font-bold text-gray-800">🏆 유사도 TOP 100 단어</h2>
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="text-gray-400 hover:text-black font-bold text-xl cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="overflow-y-auto flex-1 space-y-2 pr-1">
              {topRanks.map((item) => (
                <div
                  key={item.word}
                  className="flex justify-between items-center bg-gray-50 p-2.5 rounded-lg border text-sm"
                >
                  <span className="font-semibold text-gray-700">
                    <span className="text-gray-400 w-8 inline-block">{item.rank}위</span>
                    {item.word}
                  </span>
                  <span className="font-bold text-amber-600">{item.score}점</span>
                </div>
              ))}
            </div>

            <button
              type="button"
              onClick={() => setShowModal(false)}
              className="mt-4 w-full bg-black text-white py-2.5 rounded-xl font-bold cursor-pointer"
            >
              닫기
            </button>
          </div>
        </div>
      )}
    </main>
  );
}