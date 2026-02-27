"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { motion } from "framer-motion";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";

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

type CompareEdgeDiff = {
  index: number;
  baseline_score: number;
  candidate_score: number;
  score_delta: number;
  baseline_edge: { from_track?: string; to_track?: string; reason_code?: string };
  candidate_edge: { from_track?: string; to_track?: string; reason_code?: string };
};

type CompareResult = {
  comparison_id: string;
  baseline_run_id: string;
  candidate_run_id: string;
  delta: {
    mean_edge_score_delta: number;
    max_edge_score_delta: number;
    transition_score_delta: number;
  };
  most_improved?: CompareEdgeDiff[];
  most_regressed?: CompareEdgeDiff[];
};

type ModelStatus = {
  active_version: string | null;
  alpha: number;
  sample_count: number;
  quality_gate_thresholds?: {
    min_accuracy: number;
    max_loss: number;
  };
  promotion_thresholds?: {
    min_accuracy_delta: number;
    max_loss_delta: number;
  };
  active_quality_gate?: {
    passed: boolean;
    reasons?: string[];
  } | null;
  available_versions: Array<{
    version: string;
    sample_count?: number;
    created_at?: number;
    quality_gate?: {
      passed: boolean;
      reasons?: string[];
    };
  }>;
};

type ModelTrainingJob = {
  job_id: string;
  status: string;
  progress: number;
  result?: {
    trained?: boolean;
    version?: string;
    reason?: string;
    sample_count?: number;
    activated?: boolean;
    quality_gate?: {
      passed?: boolean;
      reasons?: string[];
    };
    promotion_gate?: {
      passed?: boolean;
      reasons?: string[];
    };
  };
  error?: string | null;
};

type ModelEvaluationReason = {
  reason_code: string;
  count: number;
};

type ModelEvaluationVersion = {
  version: string;
  sample_count: number;
  positive_ratio: number;
  rough_rate: number;
  mean_rating: number;
  dominant_reason_codes: ModelEvaluationReason[];
};

type ModelEvaluation = {
  window_days: number;
  total_labeled_feedback: number;
  active_version: string | null;
  active_metrics: ModelEvaluationVersion | null;
  versions: ModelEvaluationVersion[];
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

const sectionMotion = {
  hidden: { opacity: 0, y: 24 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.55,
      ease: [0.22, 1, 0.36, 1] as const,
      delay,
    },
  }),
};

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
  const [solverMode, setSolverMode] = useState<"classic" | "hybrid">("hybrid");
  const [beamWidth, setBeamWidth] = useState(8);
  const [annealSteps, setAnnealSteps] = useState(140);
  const [annealTempStart, setAnnealTempStart] = useState(0.08);
  const [annealTempEnd, setAnnealTempEnd] = useState(0.004);
  const [lookaheadHorizon, setLookaheadHorizon] = useState(3);
  const [lookaheadDecay, setLookaheadDecay] = useState(0.6);
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
  const [recentRuns, setRecentRuns] = useState<string[]>([]);
  const [transitionSummary, setTransitionSummary] =
    useState<TransitionDiagnostics["summary"]>();
  const [transitionDetails, setTransitionDetails] = useState<TransitionDetail[]>([]);
  const [selectedTransitionIndex, setSelectedTransitionIndex] = useState(0);
  const [compareBaselineRunId, setCompareBaselineRunId] = useState("");
  const [compareCandidateRunId, setCompareCandidateRunId] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [compareError, setCompareError] = useState("");
  const [isComparing, setIsComparing] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [modelEvaluation, setModelEvaluation] = useState<ModelEvaluation | null>(null);
  const [modelTrainingJob, setModelTrainingJob] = useState<ModelTrainingJob | null>(null);
  const [modelMessage, setModelMessage] = useState("");
  const [isTrainingModel, setIsTrainingModel] = useState(false);
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
    const controller = new AbortController();
    fetch(`${apiBase}/model/status`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (payload) {
          setModelStatus(payload as ModelStatus);
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [apiBase, result?.run_id]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/model/evaluation`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (payload) {
          setModelEvaluation(payload as ModelEvaluation);
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [apiBase, result?.run_id, modelStatus?.active_version]);

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

  useEffect(() => {
    if (!result?.run_id) {
      return;
    }
    setRecentRuns((prev) => {
      const next = [result.run_id, ...prev.filter((id) => id !== result.run_id)];
      return next.slice(0, 12);
    });
    setCompareCandidateRunId(result.run_id);
    if (!compareBaselineRunId) {
      setCompareBaselineRunId(result.run_id);
    }
  }, [result?.run_id, compareBaselineRunId]);

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
    setSolverMode("hybrid");
    setBeamWidth(8);
    setAnnealSteps(140);
    setAnnealTempStart(0.08);
    setAnnealTempEnd(0.004);
    setLookaheadHorizon(3);
    setLookaheadDecay(0.6);
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
        const safeBeamWidth = Math.min(24, Math.max(1, Math.round(beamWidth || 8)));
        const safeAnnealSteps = Math.min(1500, Math.max(0, Math.round(annealSteps || 0)));
        const safeAnnealTempStart = Math.min(
          2,
          Math.max(0.0001, annealTempStart || 0.08)
        );
        const safeAnnealTempEnd = Math.min(
          safeAnnealTempStart,
          Math.max(0.0001, annealTempEnd || 0.004)
        );
        const safeLookaheadHorizon = Math.min(
          8,
          Math.max(1, Math.round(lookaheadHorizon || 3))
        );
        const safeLookaheadDecay = Math.min(
          0.99,
          Math.max(0.05, lookaheadDecay || 0.6)
        );
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
            solver_mode: solverMode,
            beam_width: safeBeamWidth,
            anneal_steps: safeAnnealSteps,
            anneal_temp_start: safeAnnealTempStart,
            anneal_temp_end: safeAnnealTempEnd,
            lookahead_horizon: safeLookaheadHorizon,
            lookahead_decay: safeLookaheadDecay,
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

  const handleCompareRuns = async () => {
    if (!compareBaselineRunId || !compareCandidateRunId) {
      setCompareError("Pick both baseline and candidate run IDs.");
      return;
    }
    setCompareError("");
    setCompareResult(null);
    setIsComparing(true);
    try {
      const response = await fetch(`${apiBase}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          baseline_run_id: compareBaselineRunId,
          candidate_run_id: compareCandidateRunId,
          include_edge_diff: true,
          max_edges: 8,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setCompareError(payload?.detail ?? "Compare failed.");
        return;
      }
      setCompareResult(payload);
    } catch {
      setCompareError("Failed to compare runs.");
    } finally {
      setIsComparing(false);
    }
  };

  const handleTrainModel = async () => {
    setModelMessage("");
    setModelTrainingJob(null);
    setIsTrainingModel(true);
    try {
      const response = await fetch(`${apiBase}/model/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ owner_scope: "all", activate: true }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setModelMessage(payload?.detail ?? "Model training failed.");
        return;
      }
      const queuedJobId = payload?.job_id;
      if (!queuedJobId) {
        setModelMessage("Training job was not queued correctly.");
        return;
      }
      setModelTrainingJob(payload as ModelTrainingJob);
      setModelMessage("Training queued...");

      const pollDeadline = Date.now() + 120_000;
      while (Date.now() < pollDeadline) {
        const statusResponse = await fetch(`${apiBase}/model/train/${queuedJobId}`, {
          credentials: "include",
        });
        const statusPayload = (await statusResponse.json()) as ModelTrainingJob & {
          detail?: string;
        };
        if (!statusResponse.ok) {
          setModelMessage(statusPayload?.detail ?? "Failed to fetch training job status.");
          return;
        }

        setModelTrainingJob(statusPayload);
        if (statusPayload.status === "completed") {
          const resultPayload = statusPayload.result ?? {};
          if (resultPayload.trained) {
            const gateFailed = resultPayload.quality_gate?.passed === false;
            const gateReason = resultPayload.quality_gate?.reasons?.[0];
            const promotionFailed = resultPayload.promotion_gate?.passed === false;
            const promotionReason = resultPayload.promotion_gate?.reasons?.[0];
            if (resultPayload.activated) {
              setModelMessage(`Model trained + activated: ${resultPayload.version}`);
            } else if (gateFailed) {
              setModelMessage(
                gateReason
                  ? `Model trained but not activated (${gateReason}).`
                  : "Model trained but not activated (quality gate failed)."
              );
            } else if (promotionFailed) {
              setModelMessage(
                promotionReason
                  ? `Model trained but not promoted (${promotionReason}).`
                  : "Model trained but not promoted (promotion gate failed)."
              );
            } else {
              setModelMessage(`Model trained: ${resultPayload.version}`);
            }
          } else {
            setModelMessage(
              resultPayload.reason ?? "Training completed without enough labeled transitions."
            );
          }
          const refreshedStatus = await fetch(`${apiBase}/model/status`, {
            credentials: "include",
          });
          if (refreshedStatus.ok) {
            setModelStatus((await refreshedStatus.json()) as ModelStatus);
          }
          return;
        }
        if (statusPayload.status === "failed") {
          setModelMessage(statusPayload.error ?? "Model training failed.");
          return;
        }

        setModelMessage(`Training ${statusPayload.status} (${statusPayload.progress ?? 0}%)`);
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }

      setModelMessage("Training is still running. You can check status shortly.");
    } catch {
      setModelMessage("Failed to reach model training endpoint.");
    } finally {
      setIsTrainingModel(false);
    }
  };

  return (
    <motion.main
      className="app-shell"
      initial="hidden"
      animate="visible"
      variants={sectionMotion}
    >
      <div className="ambient-orb ambient-orb-left" aria-hidden="true" />
      <div className="ambient-orb ambient-orb-right" aria-hidden="true" />
      <motion.section className="hero" variants={sectionMotion} custom={0.06}>
        <div>
          <Badge variant="outline" className="hero-kicker">
            Transition-first Spotify optimizer
          </Badge>
          <h1>Mix Optimizer</h1>
          <p>
            Treat your playlist like a DJ set. The optimizer pulls tempo, key,
            and energy data to find a smoother ordering and spins up a new
            playlist with a crisp &quot;_optimized&quot; suffix.
          </p>
        </div>
        <motion.div
          whileHover={{ y: -4, scale: 1.01 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
        >
          <Card className="console-panel">
            <CardHeader>
              <CardTitle>Signal Path</CardTitle>
              <CardDescription>
                Live transition pressure across the current optimization pass.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge className="pill" variant={profile ? "default" : "secondary"}>
                {profile ? `Connected: ${profile.display_name}` : "Not connected"}
              </Badge>
              <div className="meter">
                {Array.from({ length: 12 }).map((_, index) => (
                  <motion.span
                    key={index}
                    animate={{
                      height: [12 + (index % 5) * 4, 24 + (index % 4) * 5, 16 + (index % 3) * 3],
                      opacity: [0.35, 0.95, 0.45],
                    }}
                    transition={{
                      duration: 1.9 + (index % 4) * 0.2,
                      repeat: Number.POSITIVE_INFINITY,
                      ease: "easeInOut",
                      delay: index * 0.06,
                    }}
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.section>

      <motion.section className="grid" variants={sectionMotion} custom={0.12}>
        <form className="card form-card" onSubmit={handleSubmit}>
          <Label htmlFor="playlist">Playlist URL or ID</Label>
          <Input
            id="playlist"
            type="text"
            placeholder="https://open.spotify.com/playlist/..."
            value={playlist}
            onChange={(event) => setPlaylist(event.target.value)}
            required
          />

          <Label htmlFor="name">Optional base name</Label>
          <Input
            id="name"
            type="text"
            placeholder="Late Night Switchups"
            value={mixName}
            onChange={(event) => setMixName(event.target.value)}
          />

          <Label htmlFor="preset">Preset mode</Label>
          <select
            id="preset"
            className="ui-select"
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

          <Label>Mix focus</Label>
          <div className="segmented segmented-3" role="group" aria-label="Mix focus">
              <Button
                type="button"
                variant={mixMode === "balanced" ? "default" : "outline"}
                size="sm"
                className={`seg ${mixMode === "balanced" ? "active" : ""}`}
                onClick={() => setMixMode("balanced")}
                aria-pressed={mixMode === "balanced"}
              >
                Balanced
              </Button>
              <Button
                type="button"
                variant={mixMode === "harmonic" ? "default" : "outline"}
                size="sm"
                className={`seg ${mixMode === "harmonic" ? "active" : ""}`}
                onClick={() => setMixMode("harmonic")}
                aria-pressed={mixMode === "harmonic"}
              >
                Harmonic mixing
              </Button>
              <Button
                type="button"
                variant={mixMode === "vibe" ? "default" : "outline"}
                size="sm"
                className={`seg ${mixMode === "vibe" ? "active" : ""}`}
                onClick={() => setMixMode("vibe")}
                aria-pressed={mixMode === "vibe"}
              >
                Vibe continuity
              </Button>
            </div>

          <Label>Solver mode</Label>
          <div className="segmented" role="group" aria-label="Solver mode">
            <Button
              type="button"
              variant={solverMode === "classic" ? "default" : "outline"}
              size="sm"
              className={`seg ${solverMode === "classic" ? "active" : ""}`}
              onClick={() => setSolverMode("classic")}
              aria-pressed={solverMode === "classic"}
            >
              Classic
            </Button>
            <Button
              type="button"
              variant={solverMode === "hybrid" ? "default" : "outline"}
              size="sm"
              className={`seg ${solverMode === "hybrid" ? "active" : ""}`}
              onClick={() => setSolverMode("hybrid")}
              aria-pressed={solverMode === "hybrid"}
            >
              Hybrid
            </Button>
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
              <label htmlFor="beam-width">Beam width</label>
              <input
                id="beam-width"
                type="number"
                min={1}
                max={24}
                step={1}
                value={beamWidth}
                onChange={(event) =>
                  setBeamWidth(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 8
                  )
                }
              />
            </div>

            <div>
              <label htmlFor="anneal-steps">Anneal steps</label>
              <input
                id="anneal-steps"
                type="number"
                min={0}
                max={1500}
                step={10}
                value={annealSteps}
                onChange={(event) =>
                  setAnnealSteps(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 140
                  )
                }
              />
            </div>

            <div>
              <label htmlFor="anneal-temp-start">Anneal start temp</label>
              <input
                id="anneal-temp-start"
                type="range"
                min={0.001}
                max={0.3}
                step={0.001}
                value={annealTempStart}
                onChange={(event) => setAnnealTempStart(Number(event.target.value))}
              />
              <div className="range-value">{annealTempStart.toFixed(3)}</div>
            </div>

            <div>
              <label htmlFor="anneal-temp-end">Anneal end temp</label>
              <input
                id="anneal-temp-end"
                type="range"
                min={0.001}
                max={0.1}
                step={0.001}
                value={annealTempEnd}
                onChange={(event) => setAnnealTempEnd(Number(event.target.value))}
              />
              <div className="range-value">{annealTempEnd.toFixed(3)}</div>
            </div>

            <div>
              <label htmlFor="lookahead-horizon">Lookahead horizon</label>
              <input
                id="lookahead-horizon"
                type="number"
                min={1}
                max={8}
                step={1}
                value={lookaheadHorizon}
                onChange={(event) =>
                  setLookaheadHorizon(
                    Number.isFinite(Number(event.target.value))
                      ? Number(event.target.value)
                      : 3
                  )
                }
              />
            </div>

            <div>
              <label htmlFor="lookahead-decay">Lookahead decay</label>
              <input
                id="lookahead-decay"
                type="range"
                min={0.05}
                max={0.99}
                step={0.01}
                value={lookaheadDecay}
                onChange={(event) => setLookaheadDecay(Number(event.target.value))}
              />
              <div className="range-value">{lookaheadDecay.toFixed(2)}</div>
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
              <Button type="button" variant="ghost" size="sm" onClick={resetObjective}>
                Reset objective weights
              </Button>
            </div>
            <div className="weight-grid">
              {WEIGHT_FIELDS.map((field) => (
                <div key={field.key} className="weight-row">
                  <Label htmlFor={`weight-${field.key}`}>{field.label}</Label>
                  <input
                    id={`weight-${field.key}`}
                    type="range"
                    className="ui-range"
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
              className="ui-checkbox"
              type="checkbox"
              checked={isPublic}
              onChange={(event) => setIsPublic(event.target.checked)}
            />
            <Label htmlFor="public">Make optimized playlist public</Label>
          </div>

          <div className="button-row">
            <Button type="button" variant="outline" onClick={handleConnect}>
              Connect Spotify
            </Button>
            <Button type="submit" disabled={isPending}>
              {isPending ? "Optimizing..." : "Optimize"}
            </Button>
            {profile && (
              <Button type="button" variant="ghost" onClick={handleLogout}>
                Disconnect
              </Button>
            )}
          </div>

          {status && <div className="status">{status}</div>}
          {error && <div className="status">{error}</div>}

          {result && (
            <div className="result">
              <Badge className="pill">Transition score: {result.transition_score}</Badge>
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
                      <Badge className="pill" variant="secondary" key={item.reason_code}>
                        {item.reason_code}: {item.count}
                      </Badge>
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
          <Card className="card side-card">
            <CardHeader>
              <CardTitle>How it Optimizes</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="disclaimer">
                Transitions are scored with BPM compatibility (including half/double
                time), Camelot key adjacency, loudness matching, and subtle
                energy/valence/danceability texture nudges. Choose harmonic mixing
                or vibe continuity, and optionally apply a warm-up to peak to
                cooldown energy curve. Advanced controls tune local key-locking,
                tempo ramp shaping, objective weights, and minimax passes that
                target rough edges.
              </p>
            </CardContent>
          </Card>
          <Card className="card side-card">
            <CardHeader>
              <CardTitle>A/B Compare Runs</CardTitle>
            </CardHeader>
            <CardContent>
            <Label htmlFor="compare-baseline">Baseline run</Label>
            <select
              id="compare-baseline"
              className="ui-select"
              value={compareBaselineRunId}
              onChange={(event) => setCompareBaselineRunId(event.target.value)}
            >
              <option value="">Select baseline run</option>
              {recentRuns.map((runId) => (
                <option key={`base-${runId}`} value={runId}>
                  {runId}
                </option>
              ))}
            </select>

            <Label htmlFor="compare-candidate">Candidate run</Label>
            <select
              id="compare-candidate"
              className="ui-select"
              value={compareCandidateRunId}
              onChange={(event) => setCompareCandidateRunId(event.target.value)}
            >
              <option value="">Select candidate run</option>
              {recentRuns.map((runId) => (
                <option key={`cand-${runId}`} value={runId}>
                  {runId}
                </option>
              ))}
            </select>
            <div className="button-row">
              <Button
                type="button"
                variant="outline"
                onClick={handleCompareRuns}
                disabled={isComparing}
              >
                {isComparing ? "Comparing..." : "Compare"}
              </Button>
            </div>
            {compareError && <div className="status">{compareError}</div>}
            {compareResult && (
              <div className="result">
                <div className="list">
                  <div className="list-item">
                    Mean edge delta: {compareResult.delta.mean_edge_score_delta.toFixed(4)}
                  </div>
                  <div className="list-item">
                    Max edge delta: {compareResult.delta.max_edge_score_delta.toFixed(4)}
                  </div>
                  <div className="list-item">
                    Transition delta: {compareResult.delta.transition_score_delta.toFixed(4)}
                  </div>
                </div>
                {!!compareResult.most_improved?.length && (
                  <div className="status">
                    Top improved: {compareResult.most_improved[0].baseline_edge.from_track} {"->"}{" "}
                    {compareResult.most_improved[0].baseline_edge.to_track} (
                    {compareResult.most_improved[0].score_delta.toFixed(4)})
                  </div>
                )}
                {!!compareResult.most_regressed?.length && (
                  <div className="status">
                    Top regressed: {compareResult.most_regressed[0].baseline_edge.from_track} {"->"}{" "}
                    {compareResult.most_regressed[0].baseline_edge.to_track} (
                    {compareResult.most_regressed[0].score_delta.toFixed(4)})
                  </div>
                )}
              </div>
            )}
            </CardContent>
          </Card>
          <Card className="card side-card">
            <CardHeader>
              <CardTitle>Transition Model</CardTitle>
              <CardDescription>
                Learns from explicit manual feedback labels to refine future transition
                scoring. Uses heuristic fallback when no model is active.
              </CardDescription>
            </CardHeader>
            <CardContent>
            <div className="list">
              <div className="list-item">
                Active: {modelStatus?.active_version ?? "none"}
              </div>
              <div className="list-item">
                Blend alpha: {(modelStatus?.alpha ?? 0).toFixed(2)}
              </div>
              <div className="list-item">
                Samples: {modelStatus?.sample_count ?? 0}
              </div>
              <div className="list-item">
                Gate min accuracy:{" "}
                {(modelStatus?.quality_gate_thresholds?.min_accuracy ?? 0).toFixed(2)}
              </div>
              <div className="list-item">
                Gate max loss: {(modelStatus?.quality_gate_thresholds?.max_loss ?? 0).toFixed(2)}
              </div>
              <div className="list-item">
                Promotion min acc delta:{" "}
                {(modelStatus?.promotion_thresholds?.min_accuracy_delta ?? 0).toFixed(2)}
              </div>
              <div className="list-item">
                Promotion max loss delta:{" "}
                {(modelStatus?.promotion_thresholds?.max_loss_delta ?? 0).toFixed(2)}
              </div>
              <div className="list-item">
                Active gate:{" "}
                {modelStatus?.active_quality_gate
                  ? modelStatus.active_quality_gate.passed
                    ? "pass"
                    : "fail"
                  : "n/a"}
              </div>
            </div>
            {modelEvaluation && (
              <div className="list" style={{ marginTop: 12 }}>
                <div className="list-item">
                  Feedback window: {modelEvaluation.window_days}d | labeled edges:{" "}
                  {modelEvaluation.total_labeled_feedback}
                </div>
                {modelEvaluation.active_metrics && (
                  <div className="list-item">
                    Active feedback stats: rough {modelEvaluation.active_metrics.rough_rate.toFixed(2)} |
                    mean rating {modelEvaluation.active_metrics.mean_rating.toFixed(2)}
                  </div>
                )}
                {modelEvaluation.versions.slice(0, 3).map((item) => (
                  <div className="list-item" key={`model-eval-${item.version}`}>
                    {item.version}: {item.sample_count} labels | rough {item.rough_rate.toFixed(2)} |
                    positive {item.positive_ratio.toFixed(2)}
                  </div>
                ))}
              </div>
            )}
            <div className="button-row" style={{ marginTop: 12 }}>
              <Button
                type="button"
                variant="outline"
                onClick={handleTrainModel}
                disabled={isTrainingModel}
              >
                {isTrainingModel ? "Training..." : "Train model"}
              </Button>
            </div>
            {modelTrainingJob && (
              <div className="list" style={{ marginTop: 12 }}>
                <div className="list-item">Job: {modelTrainingJob.job_id}</div>
                <div className="list-item">Status: {modelTrainingJob.status}</div>
                <div className="list-item">Progress: {modelTrainingJob.progress ?? 0}%</div>
              </div>
            )}
            {modelMessage && <div className="status">{modelMessage}</div>}
            </CardContent>
          </Card>
          <Card className="card side-card">
            <CardHeader>
              <CardTitle>Keep in Mind</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="disclaimer">
                Spotify has marked its audio-features endpoints as deprecated. This
                tool is built for personal use. Transition model training is based
                on explicit user feedback labels and transition diagnostics.
              </p>
            </CardContent>
          </Card>
        </div>
      </motion.section>
    </motion.main>
  );
}
