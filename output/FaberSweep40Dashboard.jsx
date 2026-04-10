import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, Legend, ReferenceLine, CartesianGrid
} from "recharts";

// ── Seeded PRNG ──
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normRand(rng) {
  let u = 0, v = 0;
  while (!u) u = rng();
  while (!v) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Asset colors ──
const COLORS = {
  IVV: "#2563eb", SSO: "#1d4ed8", QQQ: "#7c3aed", QLD: "#6d28d9",
  VGLT: "#059669", IAU: "#d97706", DBC: "#dc2626", Cash: "#64748b"
};
const LABELS = {
  IVV: "IVV (S&P 500)", SSO: "SSO (2× S&P)", QQQ: "QQQ (Nasdaq)",
  QLD: "QLD (2× Nasdaq)", VGLT: "VGLT (LT Treasury)", IAU: "IAU (Gold)",
  DBC: "DBC (Commodities)", Cash: "Cash"
};

// ── Market regime definitions (annualized mu/sigma per asset) ──
const REGIMES = [
  { s: [2002,0], e: [2002,9],  ivv:[-0.24,0.19], qqq:[-0.37,0.27], vglt:[0.14,0.10], iau:[0.20,0.13], dbc:[0.12,0.15] },
  { s: [2002,10],e: [2003,11], ivv:[0.24,0.14],  qqq:[0.40,0.20],  vglt:[0.01,0.10], iau:[0.20,0.12], dbc:[0.18,0.14] },
  { s: [2004,0], e: [2006,11], ivv:[0.11,0.09],  qqq:[0.08,0.13],  vglt:[0.04,0.09], iau:[0.18,0.14], dbc:[0.12,0.17] },
  { s: [2007,0], e: [2007,9],  ivv:[0.06,0.13],  qqq:[0.14,0.16],  vglt:[0.07,0.10], iau:[0.28,0.14], dbc:[0.15,0.15] },
  { s: [2007,10],e: [2009,2],  ivv:[-0.45,0.24], qqq:[-0.48,0.28], vglt:[0.20,0.14], iau:[0.08,0.17], dbc:[-0.40,0.24] },
  { s: [2009,3], e: [2009,11], ivv:[0.55,0.17],  qqq:[0.65,0.22],  vglt:[-0.12,0.12],iau:[0.12,0.14], dbc:[0.25,0.19] },
  { s: [2010,0], e: [2012,11], ivv:[0.12,0.16],  qqq:[0.15,0.18],  vglt:[0.09,0.13], iau:[0.06,0.17], dbc:[-0.04,0.17] },
  { s: [2013,0], e: [2015,11], ivv:[0.13,0.10],  qqq:[0.17,0.12],  vglt:[0.01,0.12], iau:[-0.06,0.15],dbc:[-0.15,0.15] },
  { s: [2016,0], e: [2017,11], ivv:[0.17,0.08],  qqq:[0.24,0.11],  vglt:[0.02,0.10], iau:[0.10,0.09], dbc:[0.06,0.13] },
  { s: [2018,0], e: [2018,11], ivv:[-0.06,0.15], qqq:[-0.02,0.19], vglt:[0.00,0.11], iau:[-0.03,0.10],dbc:[-0.12,0.15] },
  { s: [2019,0], e: [2019,11], ivv:[0.31,0.10],  qqq:[0.39,0.12],  vglt:[0.14,0.08], iau:[0.18,0.08], dbc:[0.06,0.10] },
  { s: [2020,0], e: [2020,3],  ivv:[-0.20,0.35], qqq:[-0.12,0.38], vglt:[0.18,0.16], iau:[0.04,0.14], dbc:[-0.35,0.28] },
  { s: [2020,4], e: [2020,11], ivv:[0.50,0.13],  qqq:[0.60,0.15],  vglt:[0.04,0.10], iau:[0.22,0.09], dbc:[0.18,0.14] },
  { s: [2021,0], e: [2021,11], ivv:[0.28,0.11],  qqq:[0.27,0.14],  vglt:[-0.06,0.11],iau:[-0.04,0.10],dbc:[0.33,0.14] },
  { s: [2022,0], e: [2022,9],  ivv:[-0.25,0.17], qqq:[-0.33,0.22], vglt:[-0.28,0.15],iau:[-0.06,0.11],dbc:[0.18,0.16] },
  { s: [2022,10],e: [2023,11], ivv:[0.20,0.12],  qqq:[0.40,0.15],  vglt:[0.03,0.11], iau:[0.14,0.10], dbc:[-0.06,0.12] },
  { s: [2024,0], e: [2025,11], ivv:[0.22,0.10],  qqq:[0.28,0.13],  vglt:[-0.03,0.09],iau:[0.12,0.09], dbc:[0.04,0.11] },
  { s: [2026,0], e: [2026,3],  ivv:[0.06,0.10],  qqq:[0.07,0.12],  vglt:[0.02,0.08], iau:[0.09,0.08], dbc:[0.03,0.10] },
];

function getRegime(y, m) {
  const v = y * 12 + m;
  for (const r of REGIMES) {
    if (v >= r.s[0]*12+r.s[1] && v <= r.e[0]*12+r.e[1]) return r;
  }
  return REGIMES[REGIMES.length - 1];
}

// ── Run full simulation ──
function runSimulation(startingCapital, monthlyContrib) {
  const rng = mulberry32(77);
  const ASSETS = ["IVV","QQQ","VGLT","IAU","DBC"];
  const KEYS = { IVV:"ivv", QQQ:"qqq", VGLT:"vglt", IAU:"iau", DBC:"dbc" };
  const BASE = { IVV:0.45, QQQ:0.25, VGLT:0.05, IAU:0.10, DBC:0.05 };
  const SUB_PCT = 0.40;
  const RFR_ANNUAL = 0.04;
  const SSO_EXP = 0.0089, QLD_EXP = 0.0095;

  const prices = {}; ASSETS.forEach(a => prices[a] = [100]);
  const dates = [];
  let mi = 0;
  for (let y = 2002; y <= 2026; y++) {
    for (let m = 0; m <= (y === 2026 ? 3 : 11); m++) {
      dates.push({ y, m, label: `${y}-${String(m+1).padStart(2,"0")}` });
      if (mi > 0) {
        const reg = getRegime(y, m);
        ASSETS.forEach(a => {
          const [mu, sig] = reg[KEYS[a]];
          const mRet = mu/12 + (sig/Math.sqrt(12)) * normRand(rng);
          prices[a].push(prices[a][prices[a].length-1] * (1 + Math.max(mRet, -0.40)));
        });
      }
      mi++;
    }
  }

  function faberScore(asset, idx) {
    let score = 0;
    for (const lb of [6, 10, 12]) {
      if (idx >= lb) {
        const slice = prices[asset].slice(idx - lb, idx);
        const sma = slice.reduce((a,b) => a+b, 0) / lb;
        if (prices[asset][idx] > sma) score++;
      } else score++;
    }
    return score;
  }

  const results = [];
  let portfolioValue = startingCapital;
  let peak = portfolioValue;
  let maxDD = 0;
  let totalContributed = startingCapital;

  for (let i = 0; i < dates.length; i++) {
    const scores = {}; ASSETS.forEach(a => scores[a] = faberScore(a, i));

    const wm = {}; ASSETS.forEach(a => {
      wm[a] = scores[a] >= 3 ? 1.0 : scores[a] === 2 ? 0.7 : 0.0;
    });

    let cashW = 0.10;
    const alloc = {};
    ASSETS.forEach(a => {
      alloc[a] = BASE[a] * wm[a];
      cashW += BASE[a] * (1 - wm[a]);
    });

    const leveraged = scores.IVV >= 3 && scores.QQQ >= 3;
    const holdings = {};
    if (leveraged) {
      const ivvBase = alloc.IVV;
      const qqqBase = alloc.QQQ;
      holdings.IVV = ivvBase * (1 - SUB_PCT);
      holdings.SSO = ivvBase * SUB_PCT;
      holdings.QQQ = qqqBase * (1 - SUB_PCT);
      holdings.QLD = qqqBase * SUB_PCT;
    } else {
      holdings.IVV = alloc.IVV;
      holdings.SSO = 0;
      holdings.QQQ = alloc.QQQ;
      holdings.QLD = 0;
    }
    holdings.VGLT = alloc.VGLT;
    holdings.IAU = alloc.IAU;
    holdings.DBC = alloc.DBC;
    holdings.Cash = cashW;

    let portRet = 0;
    if (i > 0) {
      ASSETS.forEach(a => {
        const assetRet = prices[a][i] / prices[a][i-1] - 1;
        portRet += (holdings[a] || 0) * assetRet;
        if (a === "IVV" && holdings.SSO > 0) {
          const levRet = 2 * assetRet - (RFR_ANNUAL/12) - (SSO_EXP/12);
          portRet += holdings.SSO * levRet;
        }
        if (a === "QQQ" && holdings.QLD > 0) {
          const levRet = 2 * assetRet - (RFR_ANNUAL/12) - (QLD_EXP/12);
          portRet += holdings.QLD * levRet;
        }
      });
      portRet += holdings.Cash * (RFR_ANNUAL / 12);

      portfolioValue = portfolioValue * (1 + portRet) + monthlyContrib;
      totalContributed += monthlyContrib;
    }

    peak = Math.max(peak, portfolioValue);
    const dd = (portfolioValue - peak) / peak;
    maxDD = Math.min(maxDD, dd);

    const effEquity = (holdings.IVV || 0) + (holdings.SSO || 0) * 2
      + (holdings.QQQ || 0) + (holdings.QLD || 0) * 2;

    results.push({
      idx: i,
      date: dates[i].label,
      year: dates[i].y,
      month: dates[i].m,
      portfolioValue: Math.round(portfolioValue),
      totalContributed: Math.round(totalContributed),
      gain: Math.round(portfolioValue - totalContributed),
      monthlyReturn: i > 0 ? portRet : 0,
      drawdown: dd,
      maxDD,
      leveraged,
      effEquity: Math.round(effEquity * 100),
      scores: { ...scores },
      holdings: { ...holdings },
      holdingsUSD: Object.fromEntries(
        Object.entries(holdings).map(([k,v]) => [k, Math.round(v * portfolioValue)])
      ),
    });
  }

  const monthlyRets = results.slice(1).map(r => r.monthlyReturn);
  const avgRet = monthlyRets.reduce((a,b)=>a+b,0) / monthlyRets.length;
  const stdRet = Math.sqrt(monthlyRets.reduce((a,b)=>a+(b-avgRet)**2,0) / monthlyRets.length);
  const annReturn = (1+avgRet)**12 - 1;
  const annVol = stdRet * Math.sqrt(12);
  const sharpe = annVol > 0 ? (annReturn - RFR_ANNUAL) / annVol : 0;
  const levMonths = results.filter(r => r.leveraged).length;

  return {
    results,
    stats: {
      annReturn: (annReturn * 100).toFixed(1),
      annVol: (annVol * 100).toFixed(1),
      sharpe: sharpe.toFixed(3),
      maxDD: (maxDD * 100).toFixed(1),
      finalValue: results[results.length-1].portfolioValue,
      totalContributed: results[results.length-1].totalContributed,
      totalGain: results[results.length-1].gain,
      levPct: ((levMonths / results.length) * 100).toFixed(0),
    }
  };
}

const fmtUSD = (v) => v >= 1e6 ? `$${(v/1e6).toFixed(2)}M`
  : v >= 1e3 ? `$${(v/1e3).toFixed(1)}K` : `$${v}`;

function Stat({ label, value, sub, accent }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 10, padding: "14px 16px", minWidth: 120, flex: 1
    }}>
      <div style={{ fontSize: 11, color: "#94a3b8", letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: accent || "#e2e8f0", fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function ScoreBadge({ ticker, score }) {
  const colors = ["#ef4444","#f97316","#eab308","#22c55e"];
  return (
    <div style={{ display:"inline-flex", alignItems:"center", gap:4, marginRight:12, marginBottom:4 }}>
      <span style={{ fontSize:11, color:"#94a3b8", width:32 }}>{ticker}</span>
      {[0,1,2].map(i => (
        <div key={i} style={{
          width:8, height:8, borderRadius:2,
          background: i < score ? colors[score] : "rgba(255,255,255,0.08)"
        }}/>
      ))}
    </div>
  );
}

export default function FaberSweep40Dashboard() {
  const [startCap, setStartCap] = useState(21000);
  const [monthlyC, setMonthlyC] = useState(500);
  const [timeIdx, setTimeIdx] = useState(null);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(null);

  const { results, stats } = useMemo(
    () => runSimulation(startCap, monthlyC),
    [startCap, monthlyC]
  );

  const idx = timeIdx === null ? results.length - 1 : timeIdx;
  const current = results[idx];

  useEffect(() => {
    if (playing) {
      playRef.current = setInterval(() => {
        setTimeIdx(prev => {
          const next = (prev === null ? 0 : prev) + 1;
          if (next >= results.length) { setPlaying(false); return results.length - 1; }
          return next;
        });
      }, 80);
    }
    return () => clearInterval(playRef.current);
  }, [playing, results.length]);

  const holdingEntries = Object.entries(current.holdings)
    .filter(([,v]) => v > 0.005)
    .sort((a,b) => b[1] - a[1]);

  const chartData = results.map(r => ({
    date: r.date, value: r.portfolioValue, contributed: r.totalContributed, idx: r.idx,
  }));

  const ddData = results.map(r => ({ date: r.date, dd: r.drawdown * 100, idx: r.idx }));

  return (
    <div style={{
      background: "#0f172a", color: "#e2e8f0", minHeight: "100vh",
      fontFamily: "'IBM Plex Sans', system-ui, sans-serif", padding: "20px 24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        input[type=range] { -webkit-appearance: none; background: transparent; width: 100%; }
        input[type=range]::-webkit-slider-track { height: 6px; background: #1e293b; border-radius: 3px; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: #3b82f6; border: 2px solid #0f172a; margin-top: -6px; cursor: grab; }
        input[type=range]::-webkit-slider-thumb:active { cursor: grabbing; background: #60a5fa; }
        .num-input { background: #1e293b; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #e2e8f0; padding: 8px 12px; font-size: 15px; font-family: 'JetBrains Mono', monospace; width: 130px; outline: none; }
        .num-input:focus { border-color: #3b82f6; }
      `}</style>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: -0.5 }}>
            Faber-Sweep-40
            <span style={{ color: "#3b82f6", marginLeft: 8, fontSize: 13, fontWeight: 500, verticalAlign: "middle",
              background: "rgba(59,130,246,0.15)", padding: "3px 10px", borderRadius: 20 }}>
              SIMULATION
            </span>
          </h1>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: "#64748b" }}>
            Faber SMA filter · 40% SSO/QLD substitution · freed capital → cash · monthly rebalance
          </p>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <label style={{ fontSize: 10, color: "#64748b", display: "block", marginBottom: 2, textTransform: "uppercase", letterSpacing: 0.5 }}>Starting Capital</label>
            <input className="num-input" type="number" value={startCap} min={0} step={1000}
              onChange={e => { setStartCap(+e.target.value || 0); setTimeIdx(null); }} />
          </div>
          <div>
            <label style={{ fontSize: 10, color: "#64748b", display: "block", marginBottom: 2, textTransform: "uppercase", letterSpacing: 0.5 }}>Monthly Contribution</label>
            <input className="num-input" type="number" value={monthlyC} min={0} step={100}
              onChange={e => { setMonthlyC(+e.target.value || 0); setTimeIdx(null); }} />
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        <Stat label="Portfolio Value" value={fmtUSD(current.portfolioValue)} sub={`Contributed: ${fmtUSD(current.totalContributed)}`} accent="#3b82f6" />
        <Stat label="Total Gain" value={fmtUSD(current.gain)} accent={current.gain >= 0 ? "#22c55e" : "#ef4444"} />
        <Stat label="Ann. Return" value={`${stats.annReturn}%`} accent="#a78bfa" />
        <Stat label="Sharpe" value={stats.sharpe} />
        <Stat label="Max Drawdown" value={`${stats.maxDD}%`} accent="#f97316" />
        <Stat label="Leverage" value={current.leveraged ? "ON" : "OFF"}
          sub={`${current.effEquity}% eff. equity`}
          accent={current.leveraged ? "#22c55e" : "#64748b"} />
      </div>

      <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 12, padding: "12px 16px", marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button onClick={() => { setPlaying(!playing); if (!playing && idx >= results.length-1) setTimeIdx(0); }}
              style={{ background: playing ? "#ef4444" : "#3b82f6", border: "none", color: "#fff",
                borderRadius: 6, padding: "5px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
              {playing ? "■ Stop" : "▶ Play"}
            </button>
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
              {current.date}
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 0 }}>
            {["IVV","QQQ","VGLT","IAU","DBC"].map(t => (
              <ScoreBadge key={t} ticker={t} score={current.scores[t]} />
            ))}
          </div>
        </div>
        <input type="range" min={0} max={results.length-1} value={idx}
          onChange={e => { setTimeIdx(+e.target.value); setPlaying(false); }}
          style={{ width: "100%" }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#475569", marginTop: 2 }}>
          <span>Jan 2002</span><span>2008</span><span>2014</span><span>2020</span><span>2026</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
        <div style={{ flex: 2, minWidth: 320, background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 16 }}>
          <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
            Portfolio Value
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="valGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="contGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#64748b" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#64748b" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#475569" }} tickCount={6} />
              <YAxis tick={{ fontSize: 10, fill: "#475569" }} tickFormatter={v => fmtUSD(v)} width={60} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                formatter={(v, name) => [fmtUSD(v), name === "value" ? "Portfolio" : "Contributed"]}
                labelStyle={{ color: "#94a3b8" }} />
              <Area type="monotone" dataKey="contributed" stroke="#475569" fill="url(#contGrad)" strokeWidth={1} dot={false} />
              <Area type="monotone" dataKey="value" stroke="#3b82f6" fill="url(#valGrad)" strokeWidth={2} dot={false} />
              {idx < results.length - 1 && (
                <ReferenceLine x={current.date} stroke="#f59e0b" strokeDasharray="3 3" strokeWidth={1.5} />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={{ flex: 1, minWidth: 240, background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 16 }}>
          <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
            Holdings at {current.date}
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={holdingEntries.map(([k,v]) => ({ name: k, value: Math.round(v*1000)/10 }))}
                cx="50%" cy="50%" innerRadius={45} outerRadius={75}
                dataKey="value" nameKey="name" strokeWidth={1} stroke="#0f172a">
                {holdingEntries.map(([k]) => (
                  <Cell key={k} fill={COLORS[k] || "#475569"} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                formatter={(v) => [`${v}%`]} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px", justifyContent: "center" }}>
            {holdingEntries.map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: COLORS[k] || "#475569" }} />
                <span style={{ color: "#94a3b8" }}>{k}</span>
                <span style={{ color: "#cbd5e1", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                  {(v*100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 300, background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 16 }}>
          <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
            Dollar Allocation
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={holdingEntries.map(([k,v]) => ({
              name: k, value: current.holdingsUSD[k],
              pct: (v*100).toFixed(1)
            }))} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: "#475569" }} tickFormatter={fmtUSD} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8" }} width={42} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                formatter={(v) => [fmtUSD(v)]} />
              <Bar dataKey="value" radius={[0,4,4,0]}>
                {holdingEntries.map(([k]) => <Cell key={k} fill={COLORS[k] || "#475569"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ flex: 1, minWidth: 300, background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 16 }}>
          <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
            Drawdown
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={ddData}>
              <defs>
                <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0.3} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#475569" }} tickCount={6} />
              <YAxis tick={{ fontSize: 10, fill: "#475569" }} tickFormatter={v => `${v.toFixed(0)}%`} domain={["dataMin", 0]} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                formatter={(v) => [`${v.toFixed(1)}%`]} />
              <Area type="monotone" dataKey="dd" stroke="#ef4444" fill="url(#ddGrad)" strokeWidth={1.5} dot={false} />
              {idx < results.length - 1 && (
                <ReferenceLine x={current.date} stroke="#f59e0b" strokeDasharray="3 3" strokeWidth={1.5} />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: 16, fontSize: 11, color: "#475569", textAlign: "center" }}>
        Synthetic simulation using seeded PRNG calibrated to historical market regimes · Not actual backtest results · For visualization purposes only
      </div>
    </div>
  );
}
