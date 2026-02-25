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
  run_id: string;
};

type TransitionDetail = {
  index: number;
  score: number;
  from_track: string;
  to_track: string;
  reason: string;
  reason_code: string;
  dominant_component?: string | null;
  component_share?: Record<string, number>;
  components?: Record<string, number>;
};

type TransitionDiagnostics = {
  summary?: {
    dominant_penalties?: Array<{ reason_code: string; count: number }>;
  };
  transitions?: TransitionDetail[];
};

type Profile = {
  id: string;
  display_name: string;
};

type WeightKey =
  | "bpm"
  | "key"
  | "energy"
  | "valence"
  | "dance"
  | "loudness"
  | "acousticness"
  | "instrumentalness"
  | "speechiness"
  | "liveness"
  | "time_signature";

type WeightState = Record<WeightKey, number>;

type BuiltinPreset = {
  preset_id: string;
  name: string;
  description: string;
  config: Record<string, unknown>;
};

const defaultApi = "http://localhost:8000";

const DEFAULT_WEIGHTS: WeightState = {
  bpm: 0.32,
  key: 0.28,
  energy: 0.12,
  valence: 0.06,
  dance: 0.06,
  loudness: 0.06,
  acousticness: 0.03,
  instrumentalness: 0.02,
  speechiness: 0.02,
  liveness: 0.01,
  time_signature: 0.02,
};

const FALLBACK_PRESETS: BuiltinPreset[] = [
  {
    preset_id: "warmup",
    name: "Warmup",
    description: "Gentle energy ramp with harmonic-safe transitions.",
    config: {},
  },
  {
    preset_id: "peak_hour",
    name: "Peak Hour",
    description: "Max momentum and punch with controlled cuts.",
    config: {},
  },
  {
    preset_id: "cooldown_set",
    name: "Cooldown",
    description: "Gradually lowers intensity and keeps vibe continuity.",
    config: {},
  },
  {
    preset_id: "workout",
    name: "Workout",
    description: "Stable tempo and steady energy drive.",
    config: {},
  },
  {
    preset_id: "chill",
    name: "Chill",
    description: "Low-contrast transitions with mood consistency.",
    config: {},
  },
];

const WEIGHT_FIELDS: Array<{ key: WeightKey; label: string; max: number }> = [
  { key: "bpm", label: "BPM", max: 1 },
  { key: "key", label: "Key", max: 1 },
  { key: "energy", label: "Energy", max: 1 },
  { key: "valence", label: "Valence", max: 1 },
  { key: "dance", label: "Dance", max: 1 },
  { key: "loudness", label: "Loudness", max: 1 },
  { key: "acousticness", label: "Acoustic", max: 1 },
  { key: "instrumentalness", label: "Instrumental", max: 1 },
  { key: "speechiness", label: "Speech", max: 1 },
  { key: "liveness", label: "Live", max: 1 },
  { key: "time_signature", label: "Time Signature", max: 1 },
];

export default function HomePage() {
  const apiBase = useMemo(
    () => process.env.NEXT_PUBLIC_API_BASE_URL ?? defaultApi,
    []
  );
  const [playlist, setPlaylist] = useState("");
  const [mixName, setMixName] = useState("");
  const [isPublic, setIsPublic] = useState(false);
  const [presetId, setPresetId] = useState<string>("");
  const [builtinPresets, setBuiltinPresets] =
    useState<BuiltinPreset[]>(FALLBACK_PRESETS);
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
  const [smoothnessWeight, setSmoothnessWeight] = useState(1);
  const [varietyWeight, setVarietyWeight] = useState(0);
  const [bpmWindow, setBpmWindow] = useState(0.08);
  const [maxBpmJump, setMaxBpmJump] = useState(0);
  const [minKeyCompatibility, setMinKeyCompatibility] = useState(0);
  const [noRepeatArtistWithin, setNoRepeatArtistWithin] = useState(0);
  const [weights, setWeights] = useState<WeightState>({ ...DEFAULT_WEIGHTS });
  const [profile, setProfile] = useState<Profile | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [transitionSummary, setTransitionSummary] =
    useState<TransitionDiagnostics["summary"]>();
  const [transitionDetails, setTransitionDetails] = useState<TransitionDetail[]>([]);
  const [selectedTransitionIndex, setSelectedTransitionIndex] = useState(0);
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

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/presets/builtin`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (Array.isArray(data?.items) && data.items.length > 0) {
          setBuiltinPresets(data.items as BuiltinPreset[]);
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [apiBase]);

  useEffect(() => {
    if (!result?.run_id) {
      setTransitionSummary(undefined);
      setTransitionDetails([]);
      setSelectedTransitionIndex(0);
      return;
    }
    const controller = new AbortController();
    fetch(`${apiBase}/optimize/${result.run_id}/transitions`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((payload: TransitionDiagnostics | null) => {
        if (!payload) {
          return;
        }
        setTransitionSummary(payload.summary);
        setTransitionDetails(payload.transitions ?? []);
        setSelectedTransitionIndex(0);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [apiBase, result?.run_id]);

  const handleConnect = () => {
    window.location.href = `${apiBase}/login`;
  };

  const handleLogout = () => {
    window.location.href = `${apiBase}/logout`;
  };

  const updateWeight = (key: WeightKey, value: number) => {
    setWeights((prev) => ({ ...prev, [key]: value }));
  };

  const activePreset = useMemo(
    () => builtinPresets.find((item) => item.preset_id === presetId) ?? null,
    [builtinPresets, presetId]
  );
  const selectedTransition = useMemo(
    () => transitionDetails[selectedTransitionIndex] ?? null,
    [transitionDetails, selectedTransitionIndex]
  );

  const resetObjective = () => {
    setWeights({ ...DEFAULT_WEIGHTS });
    setSmoothnessWeight(1);
    setVarietyWeight(0);
    setBpmWindow(0.08);
    setMaxBpmJump(0);
    setMinKeyCompatibility(0);
    setNoRepeatArtistWithin(0);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setResult(null);

    startTransition(async () => {
      setStatus("Optimizing transitions...");
      try {
        const safeKeyLockWindow = Math.min(
          12,
          Math.max(1, Math.round(keyLockWindow || 3))
        );
        const safeMinimaxPasses = Math.min(
          10,
          Math.max(0, Math.round(minimaxPasses || 2))
        );
        const safeTempoRampWeight = Math.min(1, Math.max(0, tempoRampWeight || 0));
        const safeSmoothnessWeight = Math.min(5, Math.max(0, smoothnessWeight || 0));
        const safeVarietyWeight = Math.min(5, Math.max(0, varietyWeight || 0));
        const safeBpmWindow = Math.min(0.5, Math.max(0.01, bpmWindow || 0.08));
        const safeMaxBpmJump = Math.min(240, Math.max(0, maxBpmJump || 0));
        const safeMinKeyCompatibility = Math.min(
          1,
          Math.max(0, minKeyCompatibility || 0)
        );
        const safeNoRepeatArtistWithin = Math.min(
          20,
          Math.max(0, Math.round(noRepeatArtistWithin || 0))
        );

        const response = await fetch(`${apiBase}/optimize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            playlist,
            name: mixName || undefined,
            public: isPublic,
            preset_id: presetId || undefined,
            mix_mode: mixMode,
            flow_curve: flowCurve,
            flow_profile: flowProfile,
            key_lock_window: safeKeyLockWindow,
            tempo_ramp_weight: safeTempoRampWeight,
            minimax_passes: safeMinimaxPasses,
            smoothness_weight: safeSmoothnessWeight,
            variety_weight: safeVarietyWeight,
            bpm_window: safeBpmWindow,
            max_bpm_jump: safeMaxBpmJump > 0 ? safeMaxBpmJump : undefined,
            min_key_compatibility:
              safeMinKeyCompatibility > 0 ? safeMinKeyCompatibility : undefined,
            no_repeat_artist_within:
              safeNoRepeatArtistWithin > 0 ? safeNoRepeatArtistWithin : 0,
            weights,
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
      } catch {
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
            playlist with a crisp &quot;_optimized&quot; suffix.
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

          <label htmlFor="preset">Preset mode</label>
          <select
            id="preset"
            value={presetId}
            onChange={(event) => setPresetId(event.target.value)}
          >
            <option value="">Manual tuning</option>
            {builtinPresets.map((preset) => (
              <option key={preset.preset_id} value={preset.preset_id}>
                {preset.name}
              </option>
            ))}
          </select>
          {activePreset && (
            <div className="status">
              Preset: <strong>{activePreset.name}</strong> — {activePreset.description}
            </div>
          )}

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
              Flow curve (warm-up {"->"} peak {"->"} cooldown)
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

            <div>
              <label htmlFor="smoothness-weight">Smoothness weight</label>
              <input
                id="smoothness-weight"
                type="range"
                min={0}
                max={5}
                step={0.1}
                value={smoothnessWeight}
                onChange={(event) => setSmoothnessWeight(Number(event.target.value))}
              />
              <div className="range-value">{smoothnessWeight.toFixed(1)}</div>
            </div>

            <div>
              <label htmlFor="variety-weight">Variety weight</label>
              <input
                id="variety-weight"
                type="range"
                min={0}
                max={5}
                step={0.1}
                value={varietyWeight}
                onChange={(event) => setVarietyWeight(Number(event.target.value))}
              />
              <div className="range-value">{varietyWeight.toFixed(1)}</div>
            </div>

            <div>
              <label htmlFor="bpm-window">BPM window</label>
              <input
                id="bpm-window"
                type="range"
                min={0.01}
                max={0.3}
                step={0.01}
                value={bpmWindow}
                onChange={(event) => setBpmWindow(Number(event.target.value))}
              />
              <div className="range-value">{bpmWindow.toFixed(2)}</div>
            </div>

            <div>
              <label htmlFor="max-bpm-jump">Max BPM jump (hard)</label>
              <input
                id="max-bpm-jump"
                type="number"
                min={0}
                max={240}
                step={1}
                value={maxBpmJump}
                onChange={(event) =>
                  setMaxBpmJump(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 0
                  )
                }
              />
            </div>

            <div>
              <label htmlFor="min-key-compatibility">Min key compatibility</label>
              <input
                id="min-key-compatibility"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={minKeyCompatibility}
                onChange={(event) => setMinKeyCompatibility(Number(event.target.value))}
              />
              <div className="range-value">{minKeyCompatibility.toFixed(2)}</div>
            </div>

            <div>
              <label htmlFor="no-repeat-artist-within">No repeat artist within</label>
              <input
                id="no-repeat-artist-within"
                type="number"
                min={0}
                max={20}
                step={1}
                value={noRepeatArtistWithin}
                onChange={(event) =>
                  setNoRepeatArtistWithin(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 0
                  )
                }
              />
            </div>
          </div>

          <div className="result" style={{ marginTop: 6 }}>
            <div className="button-row" style={{ marginBottom: 12 }}>
              <button type="button" className="secondary" onClick={resetObjective}>
                Reset objective weights
              </button>
            </div>
            <div className="weight-grid">
              {WEIGHT_FIELDS.map((field) => (
                <div key={field.key} className="weight-row">
                  <label htmlFor={`weight-${field.key}`}>{field.label}</label>
                  <input
                    id={`weight-${field.key}`}
                    type="range"
                    min={0}
                    max={field.max}
                    step={0.01}
                    value={weights[field.key]}
                    onChange={(event) => updateWeight(field.key, Number(event.target.value))}
                  />
                  <div className="range-value">{weights[field.key].toFixed(2)}</div>
                </div>
              ))}
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
                      {item.from} {"->"} {item.to} | score {item.score.toFixed(3)} | BPM{" "}
                      {item.from_bpm?.toFixed?.(1) ?? "?"} {"->"} {" "}
                      {item.to_bpm?.toFixed?.(1) ?? "?"} | Key {item.from_key} {"->"} {" "}
                      {item.to_key}
                    </div>
                  ))}
                </div>
              )}

              {transitionSummary?.dominant_penalties &&
                transitionSummary.dominant_penalties.length > 0 && (
                  <div className="dominant-reasons">
                    {transitionSummary.dominant_penalties.map((item) => (
                      <span className="pill" key={item.reason_code}>
                        {item.reason_code}: {item.count}
                      </span>
                    ))}
                  </div>
                )}

              {transitionDetails.length > 0 && (
                <div className="transition-diagnostics">
                  <div className="list">
                    {transitionDetails.slice(0, 12).map((item, index) => (
                      <button
                        type="button"
                        key={`${item.index}-${item.from_track}`}
                        className={`list-item transition-item ${
                          index === selectedTransitionIndex ? "active" : ""
                        }`}
                        onClick={() => setSelectedTransitionIndex(index)}
                      >
                        {item.from_track} {"->"} {item.to_track} | {item.reason_code} | score{" "}
                        {item.score.toFixed(3)}
                      </button>
                    ))}
                  </div>
                  {selectedTransition && (
                    <div className="transition-detail">
                      <div className="status">{selectedTransition.reason}</div>
                      <div className="weight-grid">
                        {Object.entries(selectedTransition.component_share || {}).map(
                          ([key, value]) => (
                            <div className="list-item" key={key}>
                              {key}: {(value * 100).toFixed(1)}%
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
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
              tempo ramp shaping, objective weights, and minimax passes that
              target rough edges.
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
