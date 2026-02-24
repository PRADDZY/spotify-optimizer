"use client";

import { useEffect, useMemo, useState, useTransition } from "react";

type RoughTransition = {
  score: number;
  from: string;
  to: string;
  from_bpm: number | null;
  to_bpm: number | null;
  from_key: string;
  to_key: string;
};

type OptimizeResult = {
  playlist_id: string;
  playlist_name: string;
  playlist_url: string;
  transition_score: number;
  roughest: RoughTransition[];
};

type Profile = {
  id: string;
  display_name: string;
};

const defaultApi = "http://localhost:8000";

export default function HomePage() {
  const apiBase = useMemo(
    () => process.env.NEXT_PUBLIC_API_BASE_URL ?? defaultApi,
    []
  );
  const [playlist, setPlaylist] = useState("");
  const [mixName, setMixName] = useState("");
  const [isPublic, setIsPublic] = useState(false);
  const [mixMode, setMixMode] = useState<"balanced" | "harmonic" | "vibe">(
    "harmonic"
  );
  const [flowCurve, setFlowCurve] = useState(true);
  const [flowProfile, setFlowProfile] = useState<"peak" | "gentle" | "cooldown">(
    "peak"
  );
  const [keyLockWindow, setKeyLockWindow] = useState(3);
  const [tempoRampWeight, setTempoRampWeight] = useState(0.08);
  const [minimaxPasses, setMinimaxPasses] = useState(2);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/me`, { credentials: "include", signal: controller.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.id) {
          setProfile(data);
        }
      })
      .catch(() => undefined);

    return () => controller.abort();
  }, [apiBase]);

  const handleConnect = () => {
    window.location.href = `${apiBase}/login`;
  };

  const handleLogout = () => {
    window.location.href = `${apiBase}/logout`;
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setResult(null);

    startTransition(async () => {
      setStatus("Optimizing transitions...");
      try {
        const safeKeyLockWindow = Math.min(12, Math.max(1, Math.round(keyLockWindow || 3)));
        const safeMinimaxPasses = Math.min(10, Math.max(0, Math.round(minimaxPasses || 2)));
        const safeTempoRampWeight = Math.min(1, Math.max(0, tempoRampWeight || 0));

        const response = await fetch(`${apiBase}/optimize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            playlist,
            name: mixName || undefined,
            public: isPublic,
            mix_mode: mixMode,
            flow_curve: flowCurve,
            flow_profile: flowProfile,
            key_lock_window: safeKeyLockWindow,
            tempo_ramp_weight: safeTempoRampWeight,
            minimax_passes: safeMinimaxPasses,
          }),
        });

        if (response.status === 401) {
          setStatus("");
          setError("Connect your Spotify account before optimizing.");
          return;
        }

        const payload = await response.json();
        if (!response.ok) {
          setStatus("");
          setError(payload?.detail ?? "Something went wrong.");
          return;
        }

        setResult(payload);
        setStatus("Mix optimized. Your playlist is ready.");
      } catch (err) {
        setError("Failed to reach the optimizer. Is the API running?");
      }
    });
  };

  return (
    <main>
      <section className="hero">
        <div>
          <h1>Mix Optimizer</h1>
          <p>
            Treat your playlist like a DJ set. The optimizer pulls tempo, key,
            and energy data to find a smoother ordering and spins up a new
            playlist with a crisp "_optimized" suffix.
          </p>
        </div>
        <div className="console-panel">
          <h2>Signal Path</h2>
          <div className="pill">
            {profile ? `Connected: ${profile.display_name}` : "Not connected"}
          </div>
          <div className="meter">
            {Array.from({ length: 12 }).map((_, index) => (
              <span key={index} />
            ))}
          </div>
        </div>
      </section>

      <section className="grid">
        <form className="card" onSubmit={handleSubmit}>
          <label htmlFor="playlist">Playlist URL or ID</label>
          <input
            id="playlist"
            type="text"
            placeholder="https://open.spotify.com/playlist/..."
            value={playlist}
            onChange={(event) => setPlaylist(event.target.value)}
            required
          />

          <label htmlFor="name">Optional base name</label>
          <input
            id="name"
            type="text"
            placeholder="Late Night Switchups"
            value={mixName}
            onChange={(event) => setMixName(event.target.value)}
          />

          <label>Mix focus</label>
          <div className="segmented segmented-3" role="group" aria-label="Mix focus">
            <button
              type="button"
              className={`seg ${mixMode === "balanced" ? "active" : ""}`}
              onClick={() => setMixMode("balanced")}
              aria-pressed={mixMode === "balanced"}
            >
              Balanced
            </button>
            <button
              type="button"
              className={`seg ${mixMode === "harmonic" ? "active" : ""}`}
              onClick={() => setMixMode("harmonic")}
              aria-pressed={mixMode === "harmonic"}
            >
              Harmonic mixing
            </button>
            <button
              type="button"
              className={`seg ${mixMode === "vibe" ? "active" : ""}`}
              onClick={() => setMixMode("vibe")}
              aria-pressed={mixMode === "vibe"}
            >
              Vibe continuity
            </button>
          </div>

          <div className="toggle">
            <input
              id="flow-curve"
              type="checkbox"
              checked={flowCurve}
              onChange={(event) => setFlowCurve(event.target.checked)}
            />
            <label htmlFor="flow-curve">
              Flow curve (warm-up -> peak -> cooldown)
            </label>
          </div>

          <div className="advanced-grid">
            <div>
              <label htmlFor="flow-profile">Flow profile</label>
              <select
                id="flow-profile"
                value={flowProfile}
                onChange={(event) =>
                  setFlowProfile(event.target.value as "peak" | "gentle" | "cooldown")
                }
              >
                <option value="peak">Peak</option>
                <option value="gentle">Gentle</option>
                <option value="cooldown">Cooldown</option>
              </select>
            </div>

            <div>
              <label htmlFor="key-lock-window">Key lock window</label>
              <input
                id="key-lock-window"
                type="number"
                min={1}
                max={12}
                step={1}
                value={keyLockWindow}
                onChange={(event) =>
                  setKeyLockWindow(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 3
                  )
                }
              />
            </div>

            <div>
              <label htmlFor="tempo-ramp-weight">Tempo ramp weight</label>
              <input
                id="tempo-ramp-weight"
                type="range"
                min={0}
                max={0.25}
                step={0.01}
                value={tempoRampWeight}
                onChange={(event) => setTempoRampWeight(Number(event.target.value))}
              />
              <div className="range-value">{tempoRampWeight.toFixed(2)}</div>
            </div>

            <div>
              <label htmlFor="minimax-passes">Minimax passes</label>
              <input
                id="minimax-passes"
                type="number"
                min={0}
                max={10}
                step={1}
                value={minimaxPasses}
                onChange={(event) =>
                  setMinimaxPasses(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 2
                  )
                }
              />
            </div>
          </div>

          <div className="toggle">
            <input
              id="public"
              type="checkbox"
              checked={isPublic}
              onChange={(event) => setIsPublic(event.target.checked)}
            />
            <label htmlFor="public">Make optimized playlist public</label>
          </div>

          <div className="button-row">
            <button type="button" className="secondary" onClick={handleConnect}>
              Connect Spotify
            </button>
            <button type="submit" className="primary" disabled={isPending}>
              {isPending ? "Optimizing..." : "Optimize"}
            </button>
            {profile && (
              <button type="button" className="secondary" onClick={handleLogout}>
                Disconnect
              </button>
            )}
          </div>

          {status && <div className="status">{status}</div>}
          {error && <div className="status">{error}</div>}

          {result && (
            <div className="result">
              <div className="pill">Transition score: {result.transition_score}</div>
              <div className="status">
                New playlist: <a href={result.playlist_url}>{result.playlist_name}</a>
              </div>

              {result.roughest?.length > 0 && (
                <div className="list">
                  {result.roughest.map((item, index) => (
                    <div className="list-item" key={`${item.from}-${index}`}>
                      {item.from} -> {item.to} | score {item.score.toFixed(3)} | BPM
                      {" "}
                      {item.from_bpm?.toFixed?.(1) ?? "?"} ->
                      {" "}
                      {item.to_bpm?.toFixed?.(1) ?? "?"} | Key {item.from_key} ->
                      {" "}
                      {item.to_key}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </form>

        <div className="side-panel">
          <div className="card">
            <h2 style={{ marginTop: 0 }}>How it Optimizes</h2>
            <p className="disclaimer">
              Transitions are scored with BPM compatibility (including half/double
              time), Camelot key adjacency, loudness matching, and subtle
              energy/valence/danceability texture nudges. Choose harmonic mixing
              or vibe continuity, and optionally apply a warm-up to peak to
              cooldown energy curve. Advanced controls tune local key-locking,
              tempo ramp shaping, and minimax passes that target rough edges.
            </p>
          </div>
          <div className="card">
            <h2 style={{ marginTop: 0 }}>Keep in Mind</h2>
            <p className="disclaimer">
              Spotify has marked its audio-features endpoints as deprecated. This
              tool is built for personal use and does not train any models on
              Spotify content.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}




