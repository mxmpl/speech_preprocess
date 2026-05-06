import argparse
import os

from .core import post_process_dataset, segment_dataset, subsample_dataset, vad_dataset


def cli() -> None:
    parser = argparse.ArgumentParser(prog="preprocess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_vad = subparsers.add_parser("vad", help="Run voice activity detection on a dataset")
    p_vad.add_argument("path_audios", help="Directory containing the audio files to process")
    p_vad.add_argument("path_rttm", help="Output RTTM file (appended to) for the detected speech segments")
    p_vad.add_argument(
        "--model",
        default="pyannote/segmentation-3.0",
        help="Pretrained pyannote segmentation model identifier",
    )
    p_vad.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace access token for gated models",
    )
    p_vad.add_argument(
        "--extension",
        type=str,
        default=".wav",
        help="File extension used to find audios in path_audios",
    )

    p_pp = subparsers.add_parser("post-process", help="Post-process an RTTM file")
    p_pp.add_argument("path_rttm", help="Input RTTM file to post-process")
    p_pp.add_argument("path_post_processed_rttm", help="Output RTTM file (appended to) with the post-processed turns")
    p_pp.add_argument(
        "--min-duration-on",
        type=float,
        required=True,
        help="Discard speech segments shorter than this (in seconds)",
    )
    p_pp.add_argument(
        "--min-duration-off",
        type=float,
        required=True,
        help="Merge consecutive speech segments separated by a silence shorter than this (in seconds)",
    )
    p_pp.add_argument(
        "--max-duration-on",
        type=float,
        required=True,
        help="Split segments longer than this (in seconds), using the longest silence in the original annotation",
    )

    p_sub = subparsers.add_parser("subsample", help="Subsample an RTTM file by duration distribution")
    p_sub.add_argument("path_rttm", help="Input RTTM file to subsample")
    p_sub.add_argument("path_subsampled_rttm", help="Output RTTM file with the subsampled turns")
    p_sub.add_argument(
        "--target-hours",
        type=int,
        required=True,
        help="Target total duration of the subsampled set, in hours",
    )
    p_sub.add_argument(
        "--min-duration",
        type=float,
        required=True,
        help="Lower bound (inclusive) of the duration distribution, in seconds",
    )
    p_sub.add_argument(
        "--max-duration",
        type=float,
        required=True,
        help="Upper bound (inclusive) of the duration distribution, in seconds",
    )
    p_sub.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="Number of bins used to discretize the duration distribution",
    )

    p_seg = subparsers.add_parser("segment", help="Segment audios according to an RTTM file")
    p_seg.add_argument("path_audios", help="Directory containing the source audio files")
    p_seg.add_argument("path_rttm", help="RTTM file describing the segments to extract")
    p_seg.add_argument("path_output", help="Output directory for the extracted audio segments")
    p_seg.add_argument("--num-zeros", type=int, default=5, help="Zero-padding width for the index in output filenames")
    p_seg.add_argument("--extension", default=".wav", help="File extension of the source audios and output segments")

    args = parser.parse_args()
    if args.command == "vad":
        vad_dataset(
            args.path_audios,
            args.path_rttm,
            model=args.model,
            token=args.token,
            extension=args.extension,
        )
    elif args.command == "post-process":
        post_process_dataset(
            args.path_rttm,
            args.path_post_processed_rttm,
            min_duration_on=args.min_duration_on,
            min_duration_off=args.min_duration_off,
            max_duration_on=args.max_duration_on,
        )
    elif args.command == "subsample":
        subsample_dataset(
            args.path_rttm,
            args.path_subsampled_rttm,
            target_hours=args.target_hours,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            n_bins=args.n_bins,
        )
    elif args.command == "segment":
        segment_dataset(
            args.path_audios,
            args.path_rttm,
            args.path_output,
            num_zeros=args.num_zeros,
            extension=args.extension,
        )
    else:
        parser.error("Invalid subcommand")


if __name__ == "__main__":
    cli()
