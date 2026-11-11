# Get the EXE via GitHub Actions — step by step

You have a Windows PC *and* a GitHub account, so this is the fastest route: no
toolchain setup, and the runner does the 20-minute build for you.

---

## Step 1 — create an empty repo

On GitHub: **New repository** → name it `RetroVisionAI` → **Private** is fine →
do **not** add a README or .gitignore (this folder already has them) → **Create**.

Copy the URL it shows, e.g. `https://github.com/<you>/RetroVisionAI.git`.

---

## Step 2 — push this folder

Open **Git Bash** (installs with Git for Windows) or PowerShell in this folder:

```bash
git init
git add .
git commit -m "RetroVision AI 1.0.0"
git branch -M main
git remote add origin https://github.com/<you>/RetroVisionAI.git
git push -u origin main
```

`.gitignore` is already configured, so the 455 MB of model weights, `output/`,
`temp/` and `logs/` stay out of the repo. The push is small (~10 MB).

> If Git asks who you are:
> `git config --global user.name "Your Name"` and
> `git config --global user.email "you@example.com"`

---

## Step 3 — run the build

On GitHub: **Actions** tab → **Build Windows EXE** (left sidebar) →
**Run workflow** → choose:

| Input | What you get |
|---|---|
| `cpu`  | Small build (~1.3 GB). Works on **every** Windows PC. Slower AI inference. |
| `cuda` | Big build (~2.6 GB zipped). Needs an NVIDIA GPU. Fast. |
| `both` | Both artifacts. **Default.** |

Click **Run workflow**. Takes roughly 25–40 minutes per variant (they run in
parallel).

The workflow refuses to publish a broken build: it runs the test-suite, then
verifies the frozen binary with `retrovision-cli.exe --check` (must print
`RESULT : ready`), then launches the GUI for 20 seconds to prove it does not
crash on import.

---

## Step 4 — download

When the run goes green, open it and scroll to the bottom:
**Artifacts → `RetroVisionAI-cpu-windows`** (and/or `-cuda-windows`).

Unzip → `RetroVisionAI.exe` → double-click.

If you tagged a release (`git tag v1.0.0 && git push --tags`), the same zips are
attached to the release page as proper download assets instead.

---

## Step 5 — first run on Windows

1. Launch `RetroVisionAI.exe`.
2. The app needs **FFmpeg**. Either install it
   (`winget install Gyan.FFmpeg`) or drop `ffmpeg.exe` + `ffprobe.exe` into the
   `bin\` folder next to the EXE.
3. Open **AI Model Manager** and download the model you want
   (Real-ESRGAN general-x4v3 is a good first pick, only 5 MB).
4. Dashboard → drop a video → **Enhance**.

To confirm the environment, run `retrovision-cli.exe --check` in a terminal; it
should end with `RESULT : ready`.

---

## If something fails

| Symptom | Cause / fix |
|---|---|
| Workflow fails at *Install PyTorch* | Transient PyTorch CDN issue — re-run the job. |
| Fails at *Run tests* | Real bug. Open the log, the failing test name is printed. |
| `--check` says `PyTorch : not available` | The torch wheel did not install; check the *Install PyTorch* step log. |
| `RESULT : problems found` | Usually FFmpeg missing or no models downloaded — both are expected on a fresh machine. |
| App opens then closes instantly | Run `retrovision-cli.exe --check` to see the actual error in a console. |
| `cuda` build runs but is slow | No NVIDIA GPU / old driver. Use the `cpu` build, or update the driver. |

---

## Alternative: build it yourself

If you would rather not use CI, on your Windows PC:

```powershell
powershell -ExecutionPolicy Bypass -File make_exe.ps1
```

Same output, built locally. See `BUILD-THE-EXE.md` for details and the
PyInstaller version requirement (`>=6.14` — older versions break on numpy ≥ 2.3).
