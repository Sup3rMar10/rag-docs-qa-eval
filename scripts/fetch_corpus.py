"""Download the Azure Container Apps docs at a pinned commit (reproducible, not committed to this repo).

Source: https://github.com/MicrosoftDocs/azure-docs (content licensed CC BY 4.0; see README for attribution).
Uses a sparse, shallow checkout so only articles/container-apps is fetched.
"""
import argparse
import os
import shutil
import subprocess
import tempfile

REPO = "https://github.com/MicrosoftDocs/azure-docs.git"
SUBDIR = "articles/container-apps"


def run(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--commit", default="", help="pin to this commit (default: record whatever main is today)")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp()
    try:
        run("git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1" if not a.commit else "50",
            "--sparse", REPO, tmp)
        run("git", "sparse-checkout", "set", SUBDIR, cwd=tmp)
        if a.commit:
            run("git", "fetch", "--depth", "1", "origin", a.commit, cwd=tmp)
            run("git", "checkout", a.commit, cwd=tmp)
        else:
            run("git", "checkout", cwd=tmp)
        sha = run("git", "rev-parse", "HEAD", cwd=tmp)
        src = os.path.join(tmp, SUBDIR)
        if os.path.exists(a.out):
            shutil.rmtree(a.out)
        os.makedirs(a.out)
        n = 0
        for name in sorted(os.listdir(src)):
            if name.endswith(".md"):
                shutil.copy(os.path.join(src, name), os.path.join(a.out, name))
                n += 1
        with open(os.path.join(a.out, "SOURCE.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{REPO}\ncommit: {sha}\npath: {SUBDIR}\nlicense: CC BY 4.0 (content) / MIT (code)\n")
        print(f"copied {n} markdown files at commit {sha}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
