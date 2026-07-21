# Setup — trying this out locally

This gets the project running on your own machine.

**ChatGPT App.** Runs inside a ChatGPT conversation. Uses your own ChatGPT Plus/Pro usage instead of a separate API

## 1. Prerequisites

- Python 3.10 or newer
- `pip`

## 2. Get the code

Clone the repository and navigate to the project folder

## 3. Get Cloudfare.cloudflared

Open a new terminal and run the below commands to get Cloudfare.cloudfared

**Windows (cmd):**
```bash
winget install --id Cloudflare.cloudflared
```

**macOS**

1. Download [Homebrew](https://brew.sh/)
2. Run this in temrinal:
```bash
brew install cloudflared
```

**Linux**
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo apt-get install ./cloudflared-linux-amd64.deb
```

## 4. Install dependencies

Open another new terminal (keep the cloudfare and mcp terminals different)

**Windows / macOS / Linux:**
```bash
pip install -r requirements.txt
```

## 5. Set up your `.env` file

```bash
cp .env.example .env      # macOS/Linux
copy .env.example .env    # Windows
```

Open `.env` in any text editor. Which values you need depends on which mode you pick below. Never share or commit this file — it's already listed in `.gitignore`.

## 6. Connect the plugin (ChatGPT App) to your personal ChatGPT account

**Needs:** ChatGPT Plus or Pro (Developer Mode, required to test a custom app, isn't available on the free ChatGPT tier).

1. Start the server:
   ```bash
   py mcp_server.py
   ```
2. In a **second terminal**, make it reachable over HTTPS:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   Install `cloudflared` first if you don't have it (no account needed for this)
3. It'll print a URL like `https://random-words.trycloudflare.com`. Add `/mcp` to the end — that's your server URL.
4. In ChatGPT: turn on **Developer Mode** — Settings → Apps & Connectors → Advanced settings (or Settings → Security and login, depending on your account).
5. Go to `chatgpt.com/plugins`, click **+**, paste in your `/mcp` URL, give it a name, create it.
6. In a **new chat**, ask ChatGPT to use it — use this prompt:
   > Use the Paper_Citation_Graphv1.0.0 on https://arxiv.org/abs/2401.12345, cap 30. Call fetch_paper_and_citations first, then submit_report, then show me the report_markdown field verbatim — don't write your own separate summary. Give me the result in a table.

Both terminals (server + tunnel) need to stay open while you're using it. Restarting either one gives you a new tunnel URL, so you'll need to update the app's Server URL in ChatGPT each time you restart.

---

## Optional: a Semantic Scholar API key

This is not strictly required — the plugin works on Semantic Scholar's shared free pool — but a key gets you a more reliable rate limit instead of competing with everyone else on the unauthenticated tier.

1. Apply at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) (manually reviewed, can take a few days).
2. Once approved, add it to `.env`:
   ```
   SEMANTIC_SCHOLAR_API_KEY=your-key-here
   ```

No code changes needed either way — the code picks this up automatically if it's present, and falls back gracefully if it's blank.

---

## If something goes wrong

See **README.md → "Known issues and how to read them"** for the handful of things that come up on a first run (model names changing, rate limits, needing a full restart after editing code) and what each one actually means.
