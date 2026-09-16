/**
 * V19d HSA — operational live tracker (local only).
 *
 * Pure presentation layer. It reads ONE file the Python side produces
 * (live/runtime/fidelity_tracker.json) and renders it. It fetches nothing,
 * computes no prices, and never writes portfolio state. The Python CLI is the
 * single source of truth:
 *
 *   .venv/bin/python -m live.fidelity positions <fidelity.csv>   # your holdings (CSV)
 *   .venv/bin/python -m live.fidelity refresh                    # rebuild the JSON below
 *
 * Market/benchmark prices come from yfinance inside `refresh` (the same source the
 * signals use). Your account data comes from the Fidelity CSV. No third feed.
 *
 * Dependency-free (Node >= 18 built-ins). Binds 127.0.0.1, Basic-Auth gated.
 *   node tracker/server.js        (or: npm start)
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { execFile } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const RUNTIME = path.join(ROOT, "live", "runtime");
const UPLOADS = path.join(RUNTIME, "uploads");
const DATA = path.join(__dirname, "data");
const PY = path.join(ROOT, ".venv", "bin", "python");
const PORT = Number(process.env.TRACKER_PORT || 4319);
const ORDER = ["hsa", "taxable"]; // display order in the switcher

function runPython(args) {
  return new Promise((resolve, reject) => {
    execFile(PY, ["-m", "live.fidelity", ...args], { cwd: ROOT, timeout: 120000, maxBuffer: 8e6 },
      (err, stdout, stderr) => err ? reject(new Error((stderr || err.message || "").trim())) : resolve(stdout));
  });
}

// ── auth ─────────────────────────────────────────────────────────────────────
function getPassword() {
  if (process.env.TRACKER_PASSWORD) return process.env.TRACKER_PASSWORD;
  const pwFile = path.join(DATA, "password.txt");
  if (fs.existsSync(pwFile)) return fs.readFileSync(pwFile, "utf8").trim();
  const pw = crypto.randomBytes(9).toString("base64url");
  fs.mkdirSync(DATA, { recursive: true });
  fs.writeFileSync(pwFile, pw + "\n", { mode: 0o600 });
  console.log(`\n  Generated tracker password (saved to tracker/data/password.txt):\n    ${pw}\n`);
  return pw;
}
const PASSWORD = getPassword();
const SESSIONS = new Set(); // in-memory session tokens (single user, local)

function eq(a, b) {
  const x = Buffer.from(a), y = Buffer.from(b);
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}
function parseCookies(req) {
  const out = {};
  for (const part of (req.headers.cookie || "").split(";")) {
    const i = part.indexOf("=");
    if (i > -1) out[part.slice(0, i).trim()] = part.slice(i + 1).trim();
  }
  return out;
}
function authed(req) {
  const t = parseCookies(req).sid;
  return !!t && SESSIONS.has(t);
}
function readBody(req) {
  return new Promise((resolve) => {
    let b = "";
    req.on("data", (c) => { b += c; if (b.length > 8e6) req.destroy(); });
    req.on("end", () => resolve(b));
  });
}

const LOGIN_HTML = `<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>V19d Tracker — sign in</title>
<style>body{font:15px system-ui;background:#0b1120;color:#e5e7eb;display:grid;place-items:center;height:100vh;margin:0}
form{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:28px;width:280px}
h1{font-size:16px;margin:0 0 16px}input{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #1f2937;background:#0b1120;color:#e5e7eb;font:inherit}
button{width:100%;margin-top:12px;padding:10px;border:0;border-radius:8px;background:#2563eb;color:#fff;font:inherit;font-weight:600;cursor:pointer}
.e{color:#f87171;font-size:13px;margin-top:10px;min-height:16px}</style>
<form method=POST action=/login><h1>V19d · HSA Tracker</h1>
<input type=password name=password placeholder=Password autofocus>
<button>Sign in</button><div class=e>ERR</div></form>`;

// ── data ─────────────────────────────────────────────────────────────────────
function listAccounts() {
  let found = [];
  try {
    found = fs.readdirSync(RUNTIME)
      .map((f) => (f.endsWith("_tracker.json") ? f.slice(0, -"_tracker.json".length) : null))
      .filter(Boolean);
  } catch { found = []; }
  const ordered = ORDER.filter((a) => found.includes(a));
  return [...ordered, ...found.filter((a) => !ORDER.includes(a))];
}
function readTracker(account) {
  const acct = account && /^[a-z0-9_]+$/.test(account) ? account : (listAccounts()[0] || "hsa");
  try { return JSON.parse(fs.readFileSync(path.join(RUNTIME, `${acct}_tracker.json`), "utf8")); }
  catch { return null; }
}
function readJsonl(p) {
  try { return fs.readFileSync(p, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)); }
  catch { return []; }
}

// Upload a broker CSV → Python reconciles it (broker = truth) → refresh. Python
// stays the only writer; Node just saves the file and relays the diff.
async function handleUpload(req, res) {
  let payload;
  try { payload = JSON.parse(await readBody(req)); } catch { return send(res, 400, { error: "bad JSON" }); }
  const acct = String(payload.account || "");
  if (!/^[a-z0-9_]+$/.test(acct)) return send(res, 400, { error: "bad account" });
  const content = String(payload.content || "");
  if (!content.trim()) return send(res, 400, { error: "empty file" });
  if (content.length > 8e6) return send(res, 400, { error: "file too large" });
  fs.mkdirSync(UPLOADS, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const safe = (payload.filename || "upload.csv").replace(/[^a-zA-Z0-9._-]/g, "_").slice(-60);
  const csvPath = path.join(UPLOADS, `${acct}_${stamp}_${safe}`);
  fs.writeFileSync(csvPath, content);
  try {
    const out = await runPython(["reconcile", "--account", acct, csvPath, "--json"]);
    const result = JSON.parse(out.trim().split("\n").pop());
    await runPython(["refresh", "--account", acct]).catch(() => {});
    return send(res, 200, result);
  } catch (e) {
    return send(res, 500, { error: "reconcile failed: " + String((e && e.message) || e) });
  }
}

// withdrawal preview: cash first, then holdings pro-rata (mirrors live.fidelity)
function withdrawalPreview(payload, amount) {
  if (!payload) return { error: "No tracker data yet. Run: python -m live.fidelity refresh" };
  const sleeves = payload.sleeves || {};
  const cash = Number(payload.cash || 0);
  const holdings = Object.entries(sleeves)
    .map(([s, x]) => ({ s, ...x }))
    .filter((x) => x.ticker && x.value > 0 && x.price > 0);
  const holdingsTotal = holdings.reduce((a, x) => a + x.value, 0);
  const total = cash + holdingsTotal;
  if (amount > total + 1e-6) return { error: `Only $${total.toFixed(0)} available.` };
  const fromCash = Math.min(cash, amount);
  const remainder = amount - fromCash;
  const sells = [];
  if (remainder > 0.01 && holdingsTotal > 0) {
    for (const x of holdings) {
      const raise = (remainder * x.value) / holdingsTotal;
      sells.push({ sleeve: x.s, ticker: x.ticker, shares: +(raise / x.price).toFixed(4),
        dollars: +raise.toFixed(0), price: x.price });
    }
  }
  return { amount, fromCash: +fromCash.toFixed(0), remainder: +remainder.toFixed(0),
    sells, before: +total.toFixed(0), after: +(total - amount).toFixed(0) };
}

// ── server ───────────────────────────────────────────────────────────────────
function send(res, code, body, type = "application/json") {
  res.writeHead(code, { "Content-Type": type });
  res.end(typeof body === "string" ? body : JSON.stringify(body));
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, `http://localhost:${PORT}`);

  // ── login flow (no session required) ──
  if (u.pathname === "/login" && req.method === "GET") {
    return send(res, 200, LOGIN_HTML.replace("ERR", ""), "text/html");
  }
  if (u.pathname === "/login" && req.method === "POST") {
    const body = await readBody(req);
    const pw = new URLSearchParams(body).get("password") || "";
    if (eq(pw, PASSWORD)) {
      const tok = crypto.randomBytes(24).toString("base64url");
      SESSIONS.add(tok);
      res.writeHead(302, { "Set-Cookie": `sid=${tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000`, Location: "/" });
      return res.end();
    }
    return send(res, 401, LOGIN_HTML.replace("ERR", "Wrong password"), "text/html");
  }
  if (u.pathname === "/logout") {
    const t = parseCookies(req).sid; if (t) SESSIONS.delete(t);
    res.writeHead(302, { "Set-Cookie": "sid=; Path=/; Max-Age=0", Location: "/login" });
    return res.end();
  }

  // ── everything else requires a session ──
  if (!authed(req)) {
    if (u.pathname.startsWith("/api/")) return send(res, 401, { error: "auth required" });
    res.writeHead(302, { Location: "/login" });
    return res.end();
  }

  try {
    if (u.pathname === "/api/accounts") {
      const accts = listAccounts().map((a) => {
        const t = readTracker(a);
        return { account: a, label: (t && t.label) || a };
      });
      return send(res, 200, { accounts: accts });
    }
    if (u.pathname === "/api/status") {
      const p = readTracker(u.searchParams.get("account"));
      if (!p) return send(res, 200, { seeded: false, needsRefresh: true });
      return send(res, 200, p);
    }
    if (u.pathname === "/api/withdraw") {
      const amt = Number(u.searchParams.get("amount"));
      if (!Number.isFinite(amt) || amt <= 0) return send(res, 400, { error: "amount required" });
      return send(res, 200, withdrawalPreview(readTracker(u.searchParams.get("account")), amt));
    }
    if (u.pathname === "/api/upload-csv" && req.method === "POST") {
      return handleUpload(req, res);
    }
    if (u.pathname === "/api/checkpoints") {
      const acct = (u.searchParams.get("account") || "").match(/^[a-z0-9_]+$/) ? u.searchParams.get("account") : "";
      const cps = readJsonl(path.join(RUNTIME, `${acct}_checkpoints.jsonl`)).slice(-10).reverse();
      return send(res, 200, { checkpoints: cps.map((c) => ({ ts: c.ts, source: c.source, cash: c.cash,
        n_drift: (c.diffs || []).filter((d) => d.status === "drift").length })) });
    }
    if (u.pathname === "/" || u.pathname === "/index.html") {
      return send(res, 200, fs.readFileSync(path.join(__dirname, "public", "index.html"), "utf8"), "text/html");
    }
    send(res, 404, { error: "not found" });
  } catch (e) {
    send(res, 500, { error: String((e && e.message) || e) });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`\n  V19d tracker → http://127.0.0.1:${PORT}`);
  console.log(`  Local only. Sign in with the password (TRACKER_PASSWORD env, or tracker/data/password.txt).`);
  console.log(`  Accounts: ${listAccounts().join(", ") || "(none — run: python -m live.fidelity refresh)"}\n`);
});
