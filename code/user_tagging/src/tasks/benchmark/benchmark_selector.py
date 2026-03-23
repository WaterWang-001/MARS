import json
import os

from ...core.io_helpers import load_csv


class BenchmarkSelector:
    def __init__(self, output_dir, platform_name: str, profile_bench_ids_path: str | None = None):
        self.output_dir = output_dir
        self.platform_name = platform_name
        self.force_fsync = os.getenv("BENCH_JSONL_FSYNC", "0") == "1"

        self.platform_out_dir = output_dir
        if os.path.basename(os.path.normpath(self.platform_out_dir)) != self.platform_name:
            self.platform_out_dir = os.path.join(output_dir, self.platform_name)
        os.makedirs(self.platform_out_dir, exist_ok=True)

        self.profile_output_path = os.path.join(
            self.platform_out_dir,
            f"bench_profile_{self.platform_name}.jsonl",
        )

        self.profile_bench_ids = set()
        if self.platform_name == "weibo" and profile_bench_ids_path and os.path.exists(profile_bench_ids_path):
            try:
                self.profile_bench_ids = self._load_profile_bench_ids(profile_bench_ids_path)
            except Exception as e:
                print(f"Warning: Failed to load profile bench IDs: {e}")

    def _load_profile_bench_ids(self, path: str) -> set[str]:
        path_str = str(path)
        lower = path_str.lower()

        if lower.endswith(".json"):
            with open(path_str, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return {str(uid).strip() for uid in data if str(uid).strip()}

            if isinstance(data, dict):
                for key in ("user_ids", "profile_bench_ids", "ids"):
                    value = data.get(key)
                    if isinstance(value, list):
                        return {str(uid).strip() for uid in value if str(uid).strip()}

            return set()

        df = load_csv(path_str)
        if "user_id" not in df.columns:
            return set()
        return {str(uid).strip() for uid in df["user_id"].tolist() if str(uid).strip()}

    def route_and_save(self, user_id, user_data, valid_posts, batch_date):
        uid_str = str(user_id)
        record = {
            "user_id": uid_str,
            "platform": self.platform_name,
            "batch_date": str(batch_date),
            "user_profile": {
                "name": user_data.get("display_name")
                or user_data.get("username")
                or user_data.get("screen_name")
                or user_data.get("name")
                or "",
                "description": user_data.get("bio") or user_data.get("description") or "",
                "gender": user_data.get("gender", "unknown"),
                "followers_count": user_data.get("followers_count", 0),
                "following_count": user_data.get("following_count", 0),
            },
            "posts": [
                {
                    "title": p.get("title", ""),
                    "content": p.get("content", ""),
                    "quote_content": p.get("quote_content", ""),
                    "created_at": p.get("created_at", ""),
                    "valid_tags": p.get("valid_tags", []),
                }
                for p in valid_posts
            ],
        }

        if uid_str in self.profile_bench_ids:
            self._append_jsonl(self.profile_output_path, record)
            return "profile_bench"

        safe_date_str = str(batch_date).split(" ")[0]
        daily_file_path = os.path.join(
            self.platform_out_dir,
            f"bench_{self.platform_name}_{safe_date_str}.jsonl",
        )
        self._append_jsonl(daily_file_path, record)
        return "interest_bench"

    def _append_jsonl(self, filepath, data):
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            if self.force_fsync:
                f.flush()
                os.fsync(f.fileno())
