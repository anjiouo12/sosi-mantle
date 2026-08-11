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

  // 입력창 포커스 유지를 위한 Ref
  const inputRef = useRef<HTMLInputElement>(null);

  // 접속 및 초기화 시 자동 포커스
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // 점수에 따른 색상 및 이모지 반환
  const getScoreBadge = (rank: number, score: number) => {
    if (rank === 1) return { emoji: "🎉", color: "bg-red-500", text: "text-red-500" };
    if (rank <= 10) return { emoji: "🔥", color: "bg-orange-500", text: "text-orange-500" };
    if (rank <= 100) return { emoji: "✨", color: "bg-amber-500", text: "text-amber-500" };
    if (rank <= 500) return { emoji: "💡", color: "bg-yellow-400", text: "text-yellow-600" };
    if (score > 300) return { emoji: "🌱", color: "bg-emerald-400", text: "text-emerald-600" };
    return { emoji: "❄️", color: "bg-blue-300", text: "text-blue-500" };
  };

  // 백엔드에서 TOP 100 순위 데이터 불러오기
  async function fetchTopRanks() {
    setLoadingRanks(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/top-ranks?limit=100");
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

  async function handleSubmit() {
    const trimmedInput = input.trim();

    // 입력값이 없거나 이미 로딩 중인 경우 리턴
    if (!trimmedInput || loading) return;

    // 한글 검사
    const koreanRegex = /^[가-힣]+$/;
    if (!koreanRegex.test(trimmedInput)) {
      setMessage("⚠️ 완성된 한글 단어만 입력해 주세요.");
      return;
    }

    // 중복 체크
    if (guesses.some((g) => g.word === trimmedInput)) {
      setMessage("이미 입력한 단어입니다.");
      setInput("");
      return;
    }

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/guess", {
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

      setGuesses((prev) => [...prev, newGuess].sort((a, b) => b.score - a.score));
      setInput("");

      if (result.answer) {
        setMessage("🎉 정답입니다! 축하합니다!");
        setIsGameWon(true);
      }
    } catch (error) {
      console.error(error);
      setMessage("⚠️ 서버 통신 실패 (백엔드 서버를 확인하세요)");
    } finally {
      setLoading(false);
      // 서버 요청 완료 후 포커스 재설정
      setTimeout(() => {
        inputRef.current?.focus();
      }, 50);
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-50">
      <div className="w-full max-w-lg bg-white p-8 rounded-2xl shadow-xl flex flex-col items-center">
        <h1 className="text-3xl font-extrabold mb-2 text-center text-gray-800 flex items-center gap-2">
          소시맨틀 🔥
        </h1>
        <p className="text-sm text-gray-500 mb-6 text-center">
          단어를 입력해서 정답과 얼마나 가까운지 맞춰보세요.
        </p>

        {/* 입력 폼 영역 */}
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
            disabled={loading}
            placeholder="단어 입력..."
            autoFocus
          />

          <button
            type="submit"
            disabled={loading}
            className="bg-black text-white px-6 py-3 rounded-xl font-bold hover:bg-gray-800 disabled:bg-gray-400 transition cursor-pointer disabled:cursor-not-allowed"
          >
            {loading ? "확인중" : "입력"}
          </button>
        </form>

        {/* 안내 메시지 및 정답 시 TOP 100 보기 버튼 */}
        {message && (
          <div className="mt-1 flex flex-col items-center gap-2 w-full">
            <p className="text-sm font-semibold text-center text-red-500">{message}</p>
            {isGameWon && (
              <button
                type="button"
                onClick={fetchTopRanks}
                disabled={loadingRanks}
                className="mt-1 bg-amber-500 text-white text-xs px-4 py-2.5 rounded-lg font-bold hover:bg-amber-600 transition shadow cursor-pointer"
              >
                {loadingRanks ? "불러오는 중..." : "🏆 정답 유사도 TOP 100 보기"}
              </button>
            )}
          </div>
        )}

        {/* 추측 단어 결과 리스트 */}
        <div className="mt-6 w-full space-y-3">
          {guesses.map((g) => {
            const { emoji, color, text } = getScoreBadge(g.rank, g.score);
            const progressPercent = Math.min(Math.max((g.score / 1000) * 100, 5), 100);

            return (
              <div
                key={g.word}
                className="bg-gray-50 p-3 rounded-xl border border-gray-100 flex flex-col gap-1 shadow-sm"
              >
                <div className="flex justify-between items-center text-sm font-bold">
                  <span className="flex items-center gap-1.5 text-gray-800">
                    <span>{emoji}</span>
                    <span>{g.word}</span>
                    <span className="text-xs text-gray-400 font-normal">({g.rank}위)</span>
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

      {/* 전체 순위 모달 팝업 */}
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