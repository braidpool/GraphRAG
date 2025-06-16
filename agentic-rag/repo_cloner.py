import os
import subprocess
import shutil
from pathlib import Path

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

# (1) List of Git repository URLs to clone.
#     You can also load these from a text file if you prefer.
GIT_URLS = [
    "https://github.com/sktime/sktime",
    "https://github.com/karask/python-bitcoin-utils"
]

# (2) Base directories for raw clones and for the “normalized” output:
BASE_CLONE_DIR = Path("repos")               # where repos get cloned
BASE_NORMALIZED_DIR = Path("normalized_repos")  # where filtered code goes

# (3) Directories to skip entirely (relative to each repo root). 
#     Any folder whose name matches one of these will be ignored.
IGNORED_DIR_NAMES = {
    ".git",
    "node_modules",
    "venv",
    "__pycache__",
    "build",
    "dist",
    ".idea",
    ".vscode",
    ".pytest_cache",
    # add others (e.g. “target” for Maven/Gradle, “.gradle”, etc.) as needed
}

# (4) File‐extensions to keep (i.e., source code files).
#     Adapt this list to the languages in your repos.
CODE_EXTENSIONS = {
    ".py",   # Python
    # ".java", # Java
    # ".js",   # JavaScript
    # ".ts",   # TypeScript
    # ".go",   # Go
    # ".cpp",  # C++
    # ".c",    # C
    # ".h",    # C/C++ headers
    # ".cs",   # C#
    # ".rs",   # Rust
    # ".php",  # PHP
    # ".rb",   # Ruby
}

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def clone_repos(git_urls, dest_dir):
    """
    Clone each Git URL in git_urls into dest_dir.
    If the folder already exists, skip cloning.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for url in git_urls:
        repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        repo_path = dest_dir / repo_name

        if repo_path.exists():
            print(f"[SKIP] Already cloned: {repo_name}")
            continue

        print(f"[CLONING] {url} → {repo_path}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"[OK] Cloned {repo_name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to clone {url}: {e.stderr.decode().strip()}")


def should_ignore_dir(dir_name):
    """
    Return True if dir_name is in our IGNORED_DIR_NAMES list.
    """
    return dir_name in IGNORED_DIR_NAMES


def normalize_repo(src_repo_path: Path, dst_base_path: Path):
    """
    Walk through src_repo_path, skipping any folder listed in IGNORED_DIR_NAMES.
    Copy only files whose extension is in CODE_EXTENSIONS into the parallel
    folder under dst_base_path / <repo_name> / <relative_path>.
    """
    repo_name = src_repo_path.name
    dst_repo_root = dst_base_path / repo_name
    print(f"\n[PROCESS] Normalizing: {repo_name}")
    for root, dirs, files in os.walk(src_repo_path):
        root_path = Path(root)

        # 1) Prune ignored subfolders in-place:
        #    Modify dirs[:] so that walk() won’t descend into them.
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        # Now `dirs` no longer contains any IGNORE names, so walk() skips them.

        # 2) For each file in this kept folder, check extension:
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue  # skip non‐code files

            # Compute source path and destination path:
            rel_path = root_path.relative_to(src_repo_path)
            src_file = root_path / fname
            dst_folder = dst_repo_root / rel_path
            dst_folder.mkdir(parents=True, exist_ok=True)

            dst_file = dst_folder / fname
            shutil.copy2(src_file, dst_file)
            print(f"  • Copied: {rel_path / fname}")

    print(f"[DONE] Normalized files for {repo_name} → {dst_repo_root}")


def main():
    # Step 1: Clone all repos into BASE_CLONE_DIR/
    clone_repos(GIT_URLS, BASE_CLONE_DIR)

    # Step 2: For each cloned repo, run the “normalize” pass:
    BASE_NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    for repo_dir in BASE_CLONE_DIR.iterdir():
        if not repo_dir.is_dir():
            continue
        # Normalize this repository:
        normalize_repo(repo_dir, BASE_NORMALIZED_DIR)


if __name__ == "__main__":
    main()
