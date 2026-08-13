# Atkins Consulting Renewal Intelligence — Setup and Run Guide (v2)

A plain-language companion to `app.py`, written for someone new to this. It gets
you from nothing to a working, branded, multi-page web app on your computer in
about fifteen minutes, and then, when you are ready, to a free public link anyone
can open.

This is the interactive version of the batch prototype. It regenerates the same
four-table synthetic dataset and runs Layers 1 to 3 plus the Layer-4 action-plan
placeholder, but as a clickable web app with a company identity, page-to-page
navigation, red/amber/green risk signalling, hover explanations on every column,
a full sortable account portfolio, and a single Assumptions page that carries
every caveat.

Nothing in the model changed from the batch prototype. Everything new is
presentation.

---

## Part 1 — Run it on your computer (do this once, about 15 minutes)

The steps below are written for a Mac. Windows notes are in the box at the end of
this part.

### Step 1. Install Python
Go to python.org, download the latest version, and install it, accepting the
defaults. On Windows only, tick "Add Python to PATH" on the first install screen.

### Step 2. Make a folder and put `app.py` in it
Create a folder anywhere, for example `renewal-app` on your Desktop, and put
`app.py` inside it. That is the only file you need to run it locally.

### Step 3. Open Terminal in that folder
Open the **Terminal** app (Applications → Utilities → Terminal). Type `cd `
(the letters c, d, then a space), drag the `renewal-app` folder onto the Terminal
window, and press Enter. Terminal is now working inside that folder.

### Step 4. Create a virtual environment (a clean, private sandbox)
A virtual environment keeps this project's add-ons separate from the rest of your
computer. In Terminal, type these two lines, pressing Enter after each:

```
python3 -m venv venv
source venv/bin/activate
```

After the second line your prompt shows `(venv)` at the start. That means the
sandbox is active. Each time you come back to this later, you repeat only the
`source venv/bin/activate` line.

### Step 5. Install the libraries
```
pip install streamlit numpy pandas matplotlib
```
This downloads the four free add-ons the app needs. You do this once per sandbox.
There are no new libraries compared with the earlier version, so if you set this
up before, you are already done.

### Step 6. Run the app
```
streamlit run app.py
```
Your browser opens automatically at `http://localhost:8501` showing the app.
"localhost" means it is running privately on your own machine; nothing is on the
internet yet. Leave the Terminal window open while you use the app. Closing it
stops the app.

To stop the app, click the Terminal window and press `Ctrl` + `C`.
To run it again later: open Terminal in the folder, run
`source venv/bin/activate`, then `streamlit run app.py`.

> **Windows instead of Mac.** In Step 3, open the folder, click the address bar,
> type `cmd`, and press Enter. In Step 4, the second line is
> `venv\Scripts\activate` (backslashes). Steps 5 and 6 are identical. If
> `python3` or `pip` is not found, re-open the terminal after installing Python,
> or try `python` and `pip` without the `3`.

---

## Part 2 — The pages, and what each one is for

The app has one navigation bar across the top, on every page. Click a name to
move; there is no need for the browser back button. The left sidebar holds the
scenario controls and stays available everywhere.

### The navigation bar (top of every page)
- **Home** — the landing page: what the tool does, a few live numbers, and the
  four-step method in plain language.
- **The Model** — how the tool learns the renewal path (Layers 1 and 2).
- **Portfolio** — every live account in one sortable view (Layer 3).
- **Account Plan** — the per-account hand-off to the team (Layer 4).
- **Data** — the four synthetic tables, with download buttons.
- **Assumptions** — every caveat and provisional choice, in one place.

### The sidebar (scenario controls, on every page)
Move any control and the whole app recomputes.
- **Historical accounts** — how many past accounts are used to learn the winning
  pattern. More accounts give tidier bands and a rosier picture than a real book.
- **Live accounts to score** — how many current accounts appear in the Portfolio.
  This is new in this version; raise it to make the portfolio feel fuller.
- **Healthy-account share** — the single biggest lever on the churn rate. Raise
  it to make the outcome mix look more like a healthy book.
- **Noise share** — the fraction of accounts whose outcome deliberately
  contradicts their usage. These are the accounts built to fool the model.
- **Contract term mix** — the split between 12-month and 36-month deals.
- **Segment mix** — SMB, Mid, and Enterprise shares.
- **Random seed** — the same seed reproduces the exact same data every run.

### How to read the colours (used across Portfolio and Account Plan)
Every account gets one badge:
- 🟢 **On track** — tracking the winning path; on course to renew.
- 🟡 **Needs attention** — modestly behind on at least one signal.
- 🔴 **At risk** — clearly behind (more than one full spread) on at least one
  signal; act now.

The **Current vs winning** column always spells out the direction, so there is no
guessing. "0.78 vs winning 0.94 (below target)" means the account is behind on
that signal; "1.00 vs winning 0.94 (at/above target)" means it is fine on it.

Every column header has a short hover explanation. Rest the pointer on a header
to see what the column means.

### The Portfolio page in particular
This is the command view of the whole book. Use the controls at the top to:
- **Sort by** highest risk, soonest renewal, priority, or account size.
- **Show risk levels** to filter to just the accounts you care about.
- **Open an account plan** by picking an account and clicking through.

The chart at the top plots each account by how far it is from the winning path
(vertical) against how many quarters remain to renewal (horizontal). The shaded
top-left corner is the danger zone: far off track and renewing soon. Bubble size
is the contract value, so the biggest urgent accounts are the biggest bubbles.

---

## Part 3 — Put it online for free (optional, when you are ready)

This publishes a link like `your-app.streamlit.app` that colleagues can open. No
domain and no payment required. Confirm the current free-tier terms on the site
when you do this, as they change.

1. **Make a free GitHub account** at github.com. GitHub is where the code lives so
   the hosting service can read it.
2. **Create a new repository** (a project folder on GitHub). Upload two files:
   `app.py` and a plain text file named `requirements.txt` containing exactly:
   ```
   streamlit
   numpy
   pandas
   matplotlib
   ```
   `requirements.txt` tells the host which add-ons to install, the online
   equivalent of Step 5. It is unchanged from the earlier version.
3. **Go to share.streamlit.io**, sign in with GitHub, and point it at your
   repository and `app.py`. It builds the app and gives you the public link.

Updating later: change `app.py` on GitHub and the live app rebuilds itself.

---

## Part 4 — Tips for showing this in an interview

- **Lock the look.** Pick a random seed you like and leave it. The same seed
  always produces the same accounts, so the demo looks identical every time.
- **Set the mood of the book.** For a healthy-looking portfolio, raise the
  healthy-account share to about 0.80. For a book with clear risk to talk
  through, lower it toward 0.60.
- **Fill the portfolio.** Set live accounts to score to 40 or more so the risk
  chart looks substantial.
- **Have a story ready.** Open on Home, move to The Model to show the method,
  then Portfolio to find the danger-zone accounts, then click one into its
  Account Plan. End on Assumptions to show the honesty of the approach.
- **If you publish it,** open the public link once before the interview so the
  host has already "woken up" the app and it loads instantly on the day.

---

## Part 5 — Honest limitations (unchanged from the batch prototype)

These are stated in full on the app's Assumptions page and in the design summary.
In short:
- The synthetic data is cleaner than reality; a real book is thinner per group,
  so real bands will be noisier.
- The scoring is a first-pass rule, not a trained model. It weights by account
  value, so a large under-performing account can outrank a small churning one, a
  deliberate triage choice, and adjustable.
- Only the first contract term is modelled; survivorship is handled lightly.
- Outcome bands, cohort unit, and the red/amber/green thresholds are provisional,
  matching the design summary.

None of these block the purpose, which is to prove the logic interactively before
real data exists.
