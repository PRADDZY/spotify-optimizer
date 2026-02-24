import argparse
import os

from dotenv import load_dotenv

from backend.optimizer_core import (
    create_playlist,
    get_spotify_client,
    optimize_tracks,
    optimized_name,
    parse_playlist_id,
)

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a Spotify playlist for smoother transitions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    opt = subparsers.add_parser("optimize", help="Optimize a playlist order")
    opt.add_argument("--playlist", required=True, help="Playlist URL, URI, or ID")
    opt.add_argument("--auth", choices=["user", "app"], default="user", help="Auth mode")
    opt.add_argument("--redirect-uri", default=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback"))
    opt.add_argument("--name", default="", help="Base name for the new playlist")
    opt.add_argument("--public", action="store_true", help="Create public playlist")
    opt.add_argument("--dry-run", action="store_true", help="Only compute the order, do not create playlist")
    opt.add_argument("--cache", default="cache/audio_features.json", help="Cache path for audio features")
    opt.add_argument("--missing", choices=["append", "drop"], default="append", help="How to handle tracks without key/tempo")
    opt.add_argument(
        "--mix-mode",
        choices=["balanced", "harmonic", "vibe"],
        default="balanced",
        help="Mix focus preset",
    )
    opt.add_argument("--flow-curve", action="store_true", help="Apply warm-up -> peak -> cooldown arc")
    opt.add_argument(
        "--flow-profile",
        choices=["peak", "gentle", "cooldown"],
        default="peak",
        help="Flow target shape for energy/tempo trajectory",
    )
    opt.add_argument("--key-lock-window", type=int, default=3, help="Local harmonic lock window (tracks)")
    opt.add_argument(
        "--tempo-ramp-weight",
        type=float,
        default=0.08,
        help="Weight for matching a tempo progression curve",
    )
    opt.add_argument(
        "--minimax-passes",
        type=int,
        default=2,
        help="Passes that reduce worst transition spikes",
    )
    opt.add_argument(
        "--transition-log",
        default="",
        help="Optional JSONL path to append transition diagnostics",
    )
    opt.add_argument("--restarts", type=int, default=12, help="Number of random restarts")
    opt.add_argument("--two-opt-passes", type=int, default=2, help="2-opt improvement passes")
    opt.add_argument("--seed", type=int, default=42, help="Random seed")
    opt.add_argument("--bpm-window", type=float, default=0.08, help="BPM similarity window (fraction, e.g. 0.08 = 8%%)")
    opt.add_argument("--w-bpm", type=float, default=None)
    opt.add_argument("--w-key", type=float, default=None)
    opt.add_argument("--w-energy", type=float, default=None)
    opt.add_argument("--w-valence", type=float, default=None)
    opt.add_argument("--w-dance", type=float, default=None)

    args = parser.parse_args()

    if args.command == "optimize":
        playlist_id = parse_playlist_id(args.playlist)

        scopes = "playlist-read-private playlist-read-collaborative"
        if not args.dry_run:
            scopes += " playlist-modify-private playlist-modify-public"
            if args.auth != "user":
                raise RuntimeError("Playlist creation requires --auth user")

        sp = get_spotify_client(args.auth, args.redirect_uri, scopes)

        weights = {
            k: v
            for k, v in {
                "bpm": args.w_bpm,
                "key": args.w_key,
                "energy": args.w_energy,
                "valence": args.w_valence,
                "dance": args.w_dance,
            }.items()
            if v is not None
        }

        playlist_name, ordered_tracks, cost, roughest = optimize_tracks(
            sp=sp,
            playlist_id=playlist_id,
            cache_path=args.cache,
            weights=weights,
            bpm_window=args.bpm_window,
            restarts=args.restarts,
            two_opt_passes=args.two_opt_passes,
            missing=args.missing,
            seed=args.seed,
            mix_mode=args.mix_mode,
            flow_curve=args.flow_curve,
            flow_profile=args.flow_profile,
            key_lock_window=max(1, args.key_lock_window),
            tempo_ramp_weight=max(0.0, args.tempo_ramp_weight),
            minimax_passes=max(0, args.minimax_passes),
            transition_log_path=args.transition_log or None,
        )

        print(f"Optimized transition score (lower is smoother): {cost:.3f}")
        if roughest:
            print("\nTop rough transitions:")
            for idx, item in enumerate(roughest, start=1):
                print(
                    f"{idx:02d}. {item['from']} -> {item['to']} | score={item['score']:.3f} | BPM {item['from_bpm']} -> {item['to_bpm']} | Key {item['from_key']} -> {item['to_key']}"
                )

        if args.dry_run:
            return

        base_name = args.name or playlist_name or "Playlist"
        new_name = optimized_name(base_name)

        new_id = create_playlist(sp, new_name, [t.id for t in ordered_tracks], public=args.public)
        print(f"Created new playlist: {new_name} ({new_id})")


if __name__ == "__main__":
    main()
