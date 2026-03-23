import argparse
import sys
from pathlib import Path
import shutil
import zipfile
import os
import time

# --- 1. 环境配置 ---
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 2. 导入模块 ---
# 导入模块本身 (用于修改全局配置变量)
from MARS.code.data_process import user_post         # 适配 UnifiedProcessor
from MARS.code.data_process import user_profile      # 适配 ParallelProcessor
from MARS.code.data_process import user_relationship # 适配 FastRelationshipPipeline

# 导入处理类
from MARS.code.data_process.user_post import MultiDBProcessor
from MARS.code.data_process.user_profile import ParallelProcessor
from MARS.code.data_process.user_relationship import FastRelationshipPipeline as UserRelationshipPipeline


PROJECT_ROOT = REPO_ROOT

# --- 3. 辅助函数 ---

def make_default_paths(out_dir: Path):
    """生成默认输出路径配置"""
    out_dir = Path(out_dir)
    return {
        # Post 相关 (MultiDBProcessor 输出多平台 DB)
        "post_output_dir": str(out_dir),
        "post_log_path": str(out_dir / "processed_post.log"),
        
        # Profile 相关
        "profiles_folder": str(out_dir), 
        "profiles_out": str(out_dir / "user_profiles.csv"),
        "profiles_log": str(out_dir / "processed_profiles.log"),
        
        # Relationship 相关
        "matrix_out": str(out_dir / "attention_matrix_edges.csv"),
        "scores_out": str(out_dir / "total_attention_scores.csv"),
        "edges_temp_dir": str(out_dir / "temp_edges"), 
    }

def unzip_and_flatten(input_dir: str, pattern: str = "*.zip", keep_original_zip: bool = True):
    """解压并扁平化处理"""
    src = Path(input_dir)
    if not src.exists():
        print(f"[Unzip] 目录不存在: {src}")
        return

    log_file = src / ".processed_zips.log"
    processed_set = set()

    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                processed_set = {line.strip() for line in f if line.strip()}
        except Exception:
            pass

    zip_files = list(src.glob(pattern))
    if not zip_files:
        # 如果没有zip，也不报错，可能已经是解压好的txt了
        return

    print(f"[Unzip] 发现 {len(zip_files)} 个压缩包，检查解压...")

    with open(log_file, "a", encoding="utf-8") as log_f:
        for z in zip_files:
            if z.name in processed_set:
                continue
            
            try:
                print(f"  -> Unzipping: {z.name}")
                with zipfile.ZipFile(z, "r") as zf:
                    for file_info in zf.infolist():
                        if file_info.is_dir(): continue
                        fn = Path(file_info.filename).name
                        if fn.lower().endswith(".txt") and fn:
                            dest = src / fn
                            if not dest.exists():
                                with zf.open(file_info) as source, open(dest, "wb") as target:
                                    shutil.copyfileobj(source, target)
                log_f.write(f"{z.name}\n")
                log_f.flush()
                if not keep_original_zip:
                    z.unlink()
            except Exception as e:
                print(f"  ❌ 解压出错 {z.name}: {e}")

# --- 4. 核心任务函数 ---

def run_posts(input_dir: str, output_dir: str, log_path: str):
    print(f"\n[Task: Post (Unified ETL)] Start.")
    print(f"  -> Input: {input_dir}")
    print(f"  -> DB Output Dir: {output_dir}")
    
  
    
    user_post.INPUT_DIRECTORY = Path(input_dir)
    user_post.OUTPUT_DIR = Path(output_dir)
    user_post.PROCESSED_LOG_FILE = Path(log_path)

    proc = MultiDBProcessor(input_dir=input_dir, output_dir=output_dir)
    proc.run()
    print("[Task: Post] Done.")

def run_profiles(input_dir: str, profiles_out: str, profiles_folder: str, profiles_log: str):
    print(f"\n[Task: Profile] Start.")
    print(f"  -> CSV Output: {profiles_out}")

    # 注入全局配置
    user_profile.INPUT_FOLDER = Path(input_dir)
    user_profile.OUTPUT_PATH = Path(profiles_out)
    user_profile.OUTPUT_FOLDER = Path(profiles_folder)
    user_profile.PROCESSED_FILES_LOG = Path(profiles_log)
    
    Path(profiles_folder).mkdir(parents=True, exist_ok=True)

    converter = ParallelProcessor()
    converter.run()
    print("[Task: Profile] Done.")

def run_relationships(input_dir: str, matrix_out: str, scores_out: str, temp_dir: str):
    print(f"\n[Task: Relationship] Start.")
    print(f"  -> Matrix Output: {matrix_out}")

    # 注入全局配置
    user_relationship.INPUT_DIRECTORY = Path(input_dir)
    user_relationship.OUTPUT_MATRIX_CSV = Path(matrix_out)
    user_relationship.OUTPUT_SCORES_CSV = Path(scores_out)
    user_relationship.TEMP_OUTPUT_DIR = Path(temp_dir)
    
    pipeline = UserRelationshipPipeline()
    pipeline.run()
    print("[Task: Relationship] Done.")

def run_all(input_dir: str, out_dir: str):
    start_all = time.time()
    
    # 1. 解压
    unzip_and_flatten(input_dir)
    
    # 2. 准备路径
    paths = make_default_paths(Path(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # 3. 顺序执行
    # 注意：UnifiedProcessor 不需要中间 Raw DB，它直接一步到位生成 Final DB
    run_profiles(input_dir, paths['profiles_out'], paths['profiles_folder'], paths['profiles_log'])
    run_posts(input_dir, paths['post_output_dir'], paths['post_log_path'])
    

    
    # run_relationships(input_dir, paths['matrix_out'], paths['scores_out'], paths['edges_temp_dir'])
    
    print(f"\n✅ All Pipeline Tasks Completed in {time.time() - start_all:.2f}s.")
    print(f"📂 Outputs in: {out_dir}")

# --- 5. Main 入口 ---

def main():
    parser = argparse.ArgumentParser(description="OASIS Data Processing Pipeline (Unified ETL)")
    parser.add_argument("mode", choices=["all", "posts", "profiles", "relationships"], help="Execution Mode")
    parser.add_argument("--input", required=False, help="Raw data folder (default: data/raw)")
    parser.add_argument("--out", required=False, help="Output folder (default: MARS/output)")
    
    args = parser.parse_args()

    input_dir = args.input 
    out_dir = args.out 
    paths = make_default_paths(Path(out_dir))
    
    try:
        if args.mode == "all":
            run_all(input_dir, out_dir)
            
        elif args.mode == "posts":
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            unzip_and_flatten(input_dir) 
            run_posts(input_dir, paths['post_output_dir'], paths['post_log_path'])
            
        elif args.mode == "profiles":
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            unzip_and_flatten(input_dir)
            run_profiles(input_dir, paths['profiles_out'], paths['profiles_folder'], paths['profiles_log'])
            
        elif args.mode == "relationships":
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            unzip_and_flatten(input_dir)
            run_relationships(input_dir, paths['matrix_out'], paths['scores_out'], paths['edges_temp_dir'])
            
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()