# GitHub se EXE banane ka poora tarika — step by step

Aapke paas GitHub account hai, to ye sabse aasan rasta hai. GitHub ka **Windows
computer** aapke liye build karega aur `.exe` ka zip download ke liye de dega.

**Kul time:** setup 5 min + build 25–40 min (aapko kuch karna nahi hota).

**Kya chahiye:** GitHub account + is computer par Git (ya GitHub Desktop).

---

## STEP 1 — GitHub par khali repository banao

1. Browser mein kholें: **https://github.com/new**
2. **Repository name:** `RetroVisionAI`
3. **Visibility:** `Private` chunें (ya Public — dono chalega, chhota repo hai)
4. **Initialize this repository with:** teeno checkbox **KHAALI** chhodें
   (README, .gitignore, license — kuch bhi tick na karein)
5. **Create repository** dabayein

Ab GitHub ek page dikhayega jisme `https://github.com/<aapka-naam>/RetroVisionAI.git`
likha hoga. **Yahi URL copy kar lein** — agle step mein chahiye.

> ⚠️ Agar aapne README ya .gitignore tick kar diya to push reject ho jayega.
> Us case mein: repo ko delete karke dobara banayein, ya aage diya gaya
> "force push" wala note dekhें.

---

## STEP 2 — Is folder ko repo bana kar push karein

Is folder (`RetroVisionAI`) ke andar terminal kholें:

* **Windows:** folder ke address bar mein `cmd` likh kar Enter dabayein
* Ya Git Bash kholें aur `cd` karke is folder mein jayein

Ab ye commands **ek-ek karke** chalayein:

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "RetroVision AI 1.0.0"
```

```bash
git branch -M main
```

```bash
git remote add origin https://github.com/<aapka-naam>/RetroVisionAI.git
```

> `<aapka-naam>` ki jagah apna GitHub username likhein. Copy kiya hua URL seedha
> paste karein to galti nahi hogi.

```bash
git push -u origin main
```

### Push karte waqt kya hoga

GitHub ab password accept nahi karta. Do options hain:

**Aasaan rasta — GitHub Desktop use karein:**
1. https://desktop.github.com se install karein
2. File → **Add local repository** → is folder ko chunें
3. Upar **Publish repository** dabayein → naam `RetroVisionAI` → publish

**Ya browser se login (recommended):** push karte waqt jo window khule usme
**Sign in with a browser** chunें — automatically ho jayega.

**Ya Personal Access Token:**
1. GitHub → Settings → Developer settings → Personal access tokens →
   **Tokens (classic)** → Generate new token
2. Scope: sirf `repo` tick karein
3. Token copy karein aur password ki jagah paste karein

### Push ke baad check

GitHub par apna repo refresh karein. Aapko ye dikhna chahiye:

```
app/  ai/  core/  utils/  tests/  docs/  .github/  main.py  README.md ...
```

**`models/` folder khali dikhega** — ye sahi hai. 455 MB ke model weights jaan-boojh
kar repo mein nahi hain.

---

## STEP 3 — Build chalao

1. Apne repo mein upar **Actions** tab kholein
2. Agar pehli baar hai to "Workflows aren't being run on this repository"
   ka message aa sakta hai — **I understand my workflows, go ahead** dabayein
3. Left sidebar mein **Build Windows EXE** par click karein
4. Right side mein **Run workflow** button → dropdown khulega:

   | Input | Matlab |
   |---|---|
   | `both` | **Ye chunें.** CPU + CUDA dono banega |
   | `cpu` | Sirf chhota build (~1.3 GB), har PC par chalega |
   | `cuda` | Sirf bada build (~2.6 GB), NVIDIA GPU chahiye |

5. Green **Run workflow** dabayein

> **Pehli baar `cpu` chunna behtar hai.** Ye jaldi banta hai (~25 min) aur
> har Windows machine par chalta hai. Safal hone ke baad `cuda` bana lein.

Kuch second mein page reload karein — ek naya run `yellow` (running) dikhega.
Us par click karein to live log dekh sakte hain.

---

## STEP 4 — Build ka intezaar (25–40 min)

Run ke andar aapko 17 steps dikhenge. Sabse lambे ye hain:

| Step | Kitna time | Kya karta hai |
|---|---|---|
| Install PyTorch | 5–10 min | ~2.5 GB CUDA/CPU wheel download |
| Run tests | 3–5 min | 81 tests — fail hone par build ruk jayega |
| Build EXE | 8–15 min | PyInstaller freeze |
| Zip the bundle | 3–8 min | Final zip banata hai |

Aap tab band kar sakte hain — build chalta rahega. Email bhi aa jayega.

**Green tick ✅ = safal.** Red cross ❌ = fail (neeche "agar fail ho" dekhें).

---

## STEP 5 — EXE download karein

1. Green run par click karein
2. Page ke **sabse neeche** scroll karein → **Artifacts** section
3. Download karein:
   * `RetroVisionAI-cpu-windows` — chhota, sab jagah chalega
   * `RetroVisionAI-cuda-windows` — bada, NVIDIA GPU ke liye
4. Downloaded file ek `.zip` hogi → extract karein

Andar ye milega:

```
RetroVisionAI\
├── RetroVisionAI.exe          <- DOUBLE-CLICK THIS
├── retrovision-cli.exe        <- console version (--check ke liye)
├── READ-ME-FIRST.txt
├── _internal\                 <- torch, Qt, opencv
├── models\                    <- khali (app bharega)
├── output\
├── temp\
├── logs\
├── bin\                       <- yahan ffmpeg.exe daal sakte hain
└── config.json
```

> **Note:** GitHub artifacts 30 din baad delete ho jaate hain. Build ko permanent
> rakhna ho to STEP 7 dekhें (release banayein).

---

## STEP 6 — Pehli baar chalana (Windows par)

### 6a. FFmpeg chahiye

`.exe` ko video padhne likhne ke liye FFmpeg chahiye. Do mein se ek:

**Aasaan:** PowerShell kholें aur chalayein:
```powershell
winget install --id Gyan.FFmpeg -e --accept-source-agreements
```
Phir **computer restart** karein (PATH refresh hone ke liye), ya naya terminal
kholें.

**Ya portable:** `ffmpeg.exe` aur `ffprobe.exe` download karke `bin\` folder mein
daal dein (jo EXE ke saath hi hai).

### 6b. App kholein

`RetroVisionAI.exe` par double-click karein.

**Windows SmartScreen aayega** ("Windows protected your PC") — ye normal hai
kyunki EXE signed nahi hai (code-signing certificate paid hota hai). **More info
→ Run anyway** dabayein.

### 6c. Model download karein

1. Left sidebar → **AI Model Manager**
2. Pehla model chunें: **Real-ESRGAN general x4v3** — sirf **4.7 MB**, fast
3. **Download** dabayein — official GitHub se aayega, SHA-256 verify hoga
4. 2 second mein ho jayega

> Bade models baad mein: GFPGAN (332 MB, face repair), RIFE (20 MB, frame
> interpolation), BasicVSR++ (28 MB, temporal restore).

### 6d. Pehli video enhance karein

1. **Dashboard** → video file drag karein (ya **Import Video**)
2. **Video Enhancer** khul jayega
3. **Output Resolution** → `720p` chunें
4. **Start Enhancement** dabayein
5. Output `output\` folder mein save hoga

**CPU par speed:** ~3-4 second per frame. 10 second ka 30fps video = ~5 min.
GPU (cuda build) par yahi kaam 10-20 second mein hoga.

---

## STEP 7 (optional) — Permanent release banayein

Artifacts 30 din baad mar jaate hain. Release banane se zip hamesha rahega:

```bash
git tag v1.0.0
git push --tags
```

Ab workflow khud chalega aur build ke baad zip **Releases** page par attach ho
jayega — permanent download link ke saath, jise aap kisi ko bhi bhej sakte hain.

---

## Agar build fail ho jaye

Run kholें → red step par click karein → log padhें. Sabse common:

| Log mein kya dikhe | Matlab / fix |
|---|---|
| `Install PyTorch` fail | PyTorch CDN ka temporary issue. **Re-run failed jobs** dabayein. |
| `Run tests` fail | Koi asli bug. Failing test ka naam log mein dikhega — mujhe bhej dein. |
| `--check did not report ready` | Frozen build mein koi import toota. Log bhej dein. |
| `GUI exited early` | App start hote hi crash. `retrovision-cli.exe --check` ka output bhej dein. |
| `bundle unexpectedly large` | Kuch extra collect ho gaya. |

Kisi bhi failure par run ka URL bhej dein — main log padh kar fix kar dunga.

---

## Agar push reject ho jaye

Agar aapne repo banate waqt README tick kar diya tha:

```bash
git push -u origin main --force
```

Ye GitHub ka auto-bana README hata dega. Aapka code hi rahega.

Agar error ho: `error: remote origin already exists`

```bash
git remote set-url origin https://github.com/<aapka-naam>/RetroVisionAI.git
git push -u origin main
```

---

## Short version (jo abhi chahiye)

```bash
# is folder mein
git init && git add . && git commit -m "RetroVision AI 1.0.0"
git branch -M main
git remote add origin https://github.com/<aapka-naam>/RetroVisionAI.git
git push -u origin main
```

Phir: **GitHub → Actions → Build Windows EXE → Run workflow → cpu → Run**

25 min baad: **Artifacts → RetroVisionAI-cpu-windows → download → unzip →
RetroVisionAI.exe**
