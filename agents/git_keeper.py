#!/usr/bin/env python3
"""Git-Keeper - Auto-backup SOE to GitHub"""
import os
import sys
import subprocess
from datetime import datetime

SOE_DIR = os.path.expanduser("~/soe")
GIT_REMOTE = "https://github.com/levy11vs/soe.git"  # TODO: Update

def git_commit():
    """Commit and push SOE changes"""
    os.chdir(SOE_DIR)
    
    # Check if git initialized
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", GIT_REMOTE], check=True)
    
    # Add all files
    subprocess.run(["git", "add", "-A"], check=True)
    
    # Check for changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to commit")
        return
    
    # Commit
    msg = f"SOE Auto-backup {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    subprocess.run(["git", "commit", "-m", msg], check=True)
    
    # Push (may fail if no token)
    try:
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ Pushed to GitHub")
    except:
        print("⚠️ Push failed (no token?) - commit local only")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check status
        os.chdir(SOE_DIR)
        subprocess.run(["git", "status"])
    else:
        git_commit()
