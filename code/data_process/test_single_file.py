from pathlib import Path
import sys

# Ensure repo root on sys.path for package imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MARS.code.data_process.data_process import (
    make_default_paths,
    run_posts,
    run_profiles,
)


def main():
    """Run a fixed single-file test without external arguments."""
    repo_root = Path(__file__).resolve().parents[3]
    input_dir = repo_root / "data/raw/test/single"
    out_dir = input_dir  # 输出与输入放在同一目录

    if not input_dir.exists():
        print(f"❌ Input dir missing: {input_dir}")
        return

    paths = make_default_paths(out_dir)

    # 清理旧输出（不动原始 txt）
    for key in ("post_log_path", "profiles_out", "profiles_log", "matrix_out", "scores_out"):
        p = Path(paths[key])
        if p.exists():
            p.unlink()

    for db_file in out_dir.glob("*.db"):
        db_file.unlink()

    print(f"[Test] Processing all txt in: {input_dir}")
    print(f"[Test] Outputs will be under: {out_dir}")

    run_posts(str(input_dir), paths["post_output_dir"], paths["post_log_path"])

    print("[Test] Running profile ETL...")
    run_profiles(
        str(input_dir),
        paths["profiles_out"],
        paths["profiles_folder"],
        paths["profiles_log"],
    )


if __name__ == "__main__":
    main()
