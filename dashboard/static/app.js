"use strict";
const $ = (id) => document.getElementById(id);
const STAGES = ["instruction", "reward", "check", "train", "record", "numeric", "vlm", "decision"];
const LABELS = {instruction: "指示", reward: "報酬生成", check: "検証", train: "学習", record: "録画",
                numeric: "数値判定", vlm: "AI採点", decision: "判定"};
// What each stage does (shown when a node of the flow is clicked).
const STAGE_DOC = {
  instruction: ["指示を受け取る", "人が日本語で書いた指示を受け取り、指示に合う成功条件（人が事前に承認した数値の基準）を選びます。"],
  reward: ["報酬関数を書く（LLM）", "LLM（Claude）が、指示から「報酬関数」を書きます。報酬関数は、ロボットの動きに点数を付ける Python のコードです。2周目以降は、前の周のAI採点の指摘も渡して書き直します。"],
  check: ["コードを検証する", "書かれたコードを、学習の前に偽のデータで動かし、形・数値の異常・実行エラーがないかを確かめます。失敗したら LLM に直させます（GPU の時間を無駄にしないため）。"],
  train: ["強化学習する", "強化学習（PPO）で、報酬が高くなる動きを学習します。シミュレータ（Genesis）の中で、ロボット 4096 体を同時に動かします。"],
  record: ["録画する", "学習した方策で1体を動かし、動画と、姿勢・足の高さなどの時系列を記録します。"],
  numeric: ["数値で判定する", "録画から計算した数値を、人が事前に承認した成功条件（基準の一覧）と照らし合わせます。すべての項目を満たせば合格です。"],
  vlm: ["AIが動画を見て採点する", "別のAI（VLM）が、動画のフレームと数値だけを見て、10点満点で採点します。成功条件の中身は見せません（盲検）。8点以上で、かつ「成功」と判定されたら合格です。"],
  decision: ["成功かどうかを決める", "数値判定とAI採点の「両方」が合格なら終了します。どちらかが不合格なら、AIの指摘を渡して次の周へ進みます（上限まで）。2つの判定が食い違ったときは記録します。"],
};
let S = null;            // state from the server
let exp = null;
let replay = null;       // {vt0, wall0, speed} while replaying
let offset = 0;          // server clock - client clock (live)
let selLoop = null;      // loop chosen by badge (null = latest)
let shownVideo = "";
let catalog = [];
let canRun = false;

// The public demo is a static site (scripts/export_static.py): the same screen over exported files. Nothing can be run there.
const STATIC = !!window.STATIC_SITE;
function staticFile(path) {
  const u = new URL(path, "http://x"), q = u.searchParams;
  switch (u.pathname) {
    case "/api/experiments": return "data/experiments.json";
    case "/api/state": return `data/state/${q.get("exp")}.json`;
    case "/api/trace": return `data/trace/${q.get("exp")}_${q.get("loop")}.json`;
    case "/api/overlay": return `data/overlay/${q.get("exp")}_${q.get("loop")}.json`;
    case "/api/catalog": return "data/catalog.json";
    case "/api/source": return `data/source/${q.get("stage")}__${q.get("exp")}.json`;
    default: return null;
  }
}
const videoUrl = (name) => (STATIC ? "data/video/" : "/video/") + name;

async function api(path, opts) {
  if (STATIC) {
    if (path === "/api/config") return {can_run: false};
    const f = staticFile(path);
    if (!f || (opts && opts.method === "POST")) return Promise.reject({error: "公開デモでは、学習の実行はできません"});
    const r0 = await fetch(f);
    return r0.ok ? r0.json() : Promise.reject({error: "not found"});
  }
  const r = await fetch(path, opts);
  return r.ok ? r.json() : Promise.reject(await r.json().catch(() => ({})));
}
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };

function virtualNow() {
  if (replay) return replay.vt0 + ((Date.now() - replay.wall0) / 1000) * replay.speed;
  return Date.now() / 1000 + offset;
}

// What is visible at virtual time vt: a copy of the state with later stages cut off.
function viewAt(s, vt) {   // stages that were aborted (stopped / crashed run) keep that status
  const loops = s.loops.map((lp) => {
    const stages = {};
    for (const [k, st] of Object.entries(lp.stages)) {
      if (st.start == null || st.start > vt) continue;
      stages[k] = (st.end != null && st.end <= vt) ? {...st, status: st.status === "error" || st.status === "aborted" ? st.status : "done"}
                                                  : {...st, end: null, status: "running"};
    }
    return {...lp, stages};
  }).filter((lp) => Object.keys(lp.stages).length);
  return loops;
}
const done = (lp, k) => lp.stages[k] && lp.stages[k].status === "done";

function trainFrac(lp, vt) {
  const t = lp.stages.train;
  if (!t) return 0;
  if (done(lp, "train")) return 1;
  const total = lp.train && lp.train.total;
  if (S.synthetic || replay) {
    const full = (S.loops.find((x) => x.loop === lp.loop).stages.train || {});
    return full.end > full.start ? Math.min(1, (vt - full.start) / (full.end - full.start)) : 0;
  }
  return total ? Math.min(1, (lp.train.iteration || 0) / total) : 0;
}

// The loop the flow chart follows (the running one unless the user picked a badge) and the one whose result is shown.
// Rank of a loop: a success first, then the mean of the numeric pass rate and the AI score (later loops win ties).
function loopRank(lp) {
  if (lp.decision && lp.decision.success) return 2;
  if (lp.numeric && lp.evaluation) return (lp.numeric.n_passed / lp.numeric.n_criteria + lp.evaluation.score / 10) / 2;
  return -1;
}
function bestLoop(loops) {
  let best = null;
  for (const lp of loops) if (done(lp, "record") && lp.video && loopRank(lp) >= 0 && (!best || loopRank(lp) >= loopRank(best))) best = lp;
  return best;
}

// When the run is over and nothing was picked, the best loop is shown.
function pickLoops(loops, finished) {
  const cur = loops[loops.length - 1];
  const sel = selLoop != null ? loops.find((l) => l.loop === selLoop) : null;
  const withVideo = loops.filter((lp) => done(lp, "record") && lp.video);
  const best = finished && !sel ? bestLoop(loops) : null;
  return {cur, best, flow: sel || best || cur, result: sel || best || withVideo[withVideo.length - 1] || null};
}

const videoOverride = {};   // "exp/loop" -> the video that the overlay data belongs to (a re-recording)
let ov = null, ovKey = "", ovOn = true, ovFocus = null;   // overlay data of the loop on screen, on/off, joint under the pointer

async function ensureOverlay(result) {
  const key = `${exp}/${result.loop}`;
  if (ovKey === key) return;
  if (ovFailAt[key] && Date.now() - ovFailAt[key] < 2500) return;
  ovKey = key; ov = null; updateOvBtn();
  try {
    const o = await api(`/api/overlay?exp=${encodeURIComponent(exp)}&loop=${result.loop}`);
    if (ovKey !== key) return;
    cleanOverlay(o);
    ov = o;
    if (o.video && o.video !== result.video && videoOverride[key] !== o.video) { videoOverride[key] = o.video; shownVideo = ""; }
  } catch (e) { if (ovKey === key) { ov = null; if (e && e.error !== "not found") { ovFailAt[key] = Date.now(); ovKey = ""; } } }   // not saved is final; anything else is retried
  updateOvBtn();
}
const ovFailAt = {};
// Values the simulation could not give (null) are filled from the previous step, so that the camera and the skeleton stay defined.
function cleanOverlay(o) {
  let last = [0, 0];
  o.base_xy = (o.base_xy || []).map((p) => { if (Array.isArray(p) && p.every(Number.isFinite)) last = p; return last; });
  let prev = null;
  o.pts = (o.pts || []).map((step) => { const ok = Array.isArray(step) && step.every((q) => Array.isArray(q) && q.every(Number.isFinite)); if (ok) prev = step; return ok ? step : (prev || step); });
}
// ---- overlay tools on the video: joints + path, axes with scale, and a free camera (a 3D view of the skeleton) ----
let ovScale = false, viewFree = false;
function updateOvBtn() {
  const have = !!ov;
  const set = (id, text, on, tip) => { const b = $(id); b.hidden = false; b.disabled = !have; b.textContent = text; b.title = have ? tip : "この録画には、関節の位置が保存されていません（新しく録画した周には保存されます）"; b.classList.toggle("off", !have || !on); };
  set("ovbtn", have ? `関節と軌跡: ${ovOn ? "オン" : "オフ"}` : "関節と軌跡: 未保存", ovOn, "ロボットの関節の位置と、移動した軌跡を、動画に重ねて表示します（キー J）");
  set("scalebtn", `目盛り: ${ovScale ? "オン" : "オフ"}`, ovScale, "座標の目盛りを表示します。地面の格子（1 m）は開始位置が原点で、グラフの x_m・y_m と同じ値。縦の定規は、足や頭の高さ（グラフの height / foot_z / head_z）と同じ値（キー G）");
  set("viewbtn", viewFree ? "視点: 自由（骨格の3D）" : "視点: 動画", viewFree, "動画の視点は変えられません。代わりに、骨格を3Dで描き、ドラッグで回転・ホイールで拡大縮小できます（キー V）");
  $("freectl").hidden = !(have && viewFree);
  $("video").style.visibility = have && viewFree ? "hidden" : "visible";
  $("overlay").style.pointerEvents = have && viewFree ? "auto" : "none";
  $("overlay").style.cursor = have && viewFree ? "grab" : "";
}

const LEG_COLOR = {FR: "#3b82f6", FL: "#ef4444", RR: "#22c55e", RL: "#a855f7"};
const AXIS_COLOR = {x: "#e5484d", y: "#3fb950", z: "#58a6ff"};
const FREE0 = {az: -Math.PI / 2, el: 0.3, dist: 2.7};
let fc = {...FREE0}, fdrag = null;

// ---- a small pinhole camera (world: x, y on the ground, z up) ----
const v3 = {sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
            cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
            norm: (a) => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; }};
function makeCam(c, t, fovDeg, aspect) {
  const f = v3.norm(v3.sub(t, c)), r = v3.norm(v3.cross(f, [0, 0, 1])), u = v3.cross(r, f);
  return {c, f, r, u, th: Math.tan((fovDeg * Math.PI) / 360), aspect};
}
// The recorded camera: it follows the trunk (see scripts/record.py).
function recordedCam(o, step) {
  const [bx, by] = o.base_xy[Math.min(step, o.base_xy.length - 1)], off = o.cam_offset;
  return makeCam([bx + off[0], by + off[1], off[2]], [bx, by, o.lookat_z], o.fov, o.res[0] / o.res[1]);
}
function freeCam(o, step, aspect) {
  const [bx, by] = o.base_xy[Math.min(step, o.base_xy.length - 1)], t = [bx, by, 0.35];
  const ce = Math.cos(fc.el);
  return makeCam([t[0] + fc.dist * ce * Math.cos(fc.az), t[1] + fc.dist * ce * Math.sin(fc.az), t[2] + fc.dist * Math.sin(fc.el)], t, 42, aspect);
}
const viewOf = (cam, p) => { const d = v3.sub(p, cam.c); return [v3.dot(d, cam.r), v3.dot(d, cam.u), v3.dot(d, cam.f)]; };
const toScr = (cam, v, W, H) => [(v[0] / (v[2] * cam.th * cam.aspect) + 1) / 2 * W, (1 - v[1] / (v[2] * cam.th)) / 2 * H];
function proj(cam, p, W, H) { const v = viewOf(cam, p); return v[2] <= 0.05 ? null : toScr(cam, v, W, H); }
function projSeg(cam, a, b, W, H) {   // a segment clipped at the near plane
  let va = viewOf(cam, a), vb = viewOf(cam, b); const n = 0.05;
  if (va[2] < n && vb[2] < n) return null;
  if (va[2] < n) { const k = (n - va[2]) / (vb[2] - va[2]); va = va.map((x, i) => x + (vb[i] - x) * k); }
  if (vb[2] < n) { const k = (n - vb[2]) / (va[2] - vb[2]); vb = vb.map((x, i) => x + (va[i] - x) * k); }
  return [toScr(cam, va, W, H), toScr(cam, vb, W, H)];
}

const nice = (span) => { const raw = span / 4, p = Math.pow(10, Math.floor(Math.log10(raw))); return [1, 2, 5, 10].map((m) => m * p).find((x) => x >= raw); };

// Ground grid (origin = the start position, so x / y match the graphs' x_m / y_m), and a height ruler.
function drawGrid(g, cam, o, step, W, H, ox, oy, labels) {
  const [x0, y0] = o.base_xy[0], [bx, by] = o.base_xy[Math.min(step, o.base_xy.length - 1)], R = 3.0, st = 0.5;
  g.lineWidth = 1; g.font = "11px system-ui, sans-serif"; g.textBaseline = "middle";
  const line = (a, b, major) => {
    const s = projSeg(cam, a, b, W, H); if (!s) return;
    g.strokeStyle = `rgba(200,215,235,${major ? 0.3 : 0.14})`; g.beginPath(); g.moveTo(ox + s[0][0], oy + s[0][1]); g.lineTo(ox + s[1][0], oy + s[1][1]); g.stroke();
  };
  const kx0 = Math.ceil((bx - R - x0) / st), kx1 = Math.floor((bx + R - x0) / st), ky0 = Math.ceil((by - R - y0) / st), ky1 = Math.floor((by + R - y0) / st);
  for (let k = kx0; k <= kx1; k++) line([x0 + k * st, by - R, 0], [x0 + k * st, by + R, 0], k % 2 === 0);
  for (let k = ky0; k <= ky1; k++) line([bx - R, y0 + k * st, 0], [bx + R, y0 + k * st, 0], k % 2 === 0);
  if (!labels) return;
  g.fillStyle = "rgba(235,240,250,.85)"; g.textAlign = "center";
  for (let k = kx0; k <= kx1; k++) if (k % 2 === 0) { const q = proj(cam, [x0 + k * st, by - 1.2, 0], W, H); if (q) g.fillText(`x ${(k * st).toFixed(0)}`, ox + q[0], oy + q[1]); }
  for (let k = ky0; k <= ky1; k++) if (k % 2 === 0) { const q = proj(cam, [bx - 1.2, y0 + k * st, 0], W, H); if (q) g.fillText(`y ${(k * st).toFixed(0)}`, ox + q[0], oy + q[1]); }
  // height ruler, beside the robot (0 .. 1 m)
  const rh = v3.norm([cam.r[0], cam.r[1], 0]), rx = bx + rh[0] * 0.55, ry = by + rh[1] * 0.55;
  const top = proj(cam, [rx, ry, 1.0], W, H), bot = proj(cam, [rx, ry, 0], W, H);
  if (top && bot) {
    g.strokeStyle = "rgba(235,240,250,.8)"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(ox + bot[0], oy + bot[1]); g.lineTo(ox + top[0], oy + top[1]); g.stroke();
    g.textAlign = "left"; g.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const q = proj(cam, [rx, ry, i / 10], W, H), t2 = proj(cam, [rx + rh[0] * 0.04, ry + rh[1] * 0.04, i / 10], W, H);
      if (!q || !t2) continue;
      g.beginPath(); g.moveTo(ox + q[0], oy + q[1]); g.lineTo(ox + t2[0] + (i % 2 ? 0 : 3), oy + t2[1]); g.stroke();
      if (i % 2 === 0) { g.fillStyle = "rgba(235,240,250,.9)"; g.fillText(`${(i / 10).toFixed(1)} m`, ox + t2[0] + 6, oy + t2[1]); }
    }
  }
}

// A small axes gizmo: the directions of the world x / y / z as seen from the current camera.
function drawAxes(g, cam, cx, cy) {
  const len = 30; g.lineWidth = 2.5; g.font = "bold 12px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  const dirs = [["x", [1, 0, 0]], ["y", [0, 1, 0]], ["z", [0, 0, 1]]].map(([n, d]) => [n, v3.dot(d, cam.r), -v3.dot(d, cam.u), v3.dot(d, cam.f)]).sort((a, b) => b[3] - a[3]);
  g.fillStyle = "rgba(15,18,25,.6)"; g.beginPath(); g.arc(cx, cy, len + 14, 0, 7); g.fill();
  for (const [n, dx, dy] of dirs) {
    g.strokeStyle = g.fillStyle = AXIS_COLOR[n];
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + dx * len, cy + dy * len); g.stroke();
    g.fillText(n, cx + dx * (len + 9), cy + dy * (len + 9));
  }
}

function skeletonPoints(o, step, cam, W, H, ox, oy) {
  const pts = {};
  o.names.forEach((n, i) => { const q = proj(cam, o.pts[step][i], W, H); if (q) pts[n] = [ox + q[0], oy + q[1]]; });
  return pts;
}

function drawTrail(g, o, step, cam, W, H, ox, oy) {
  const n = o.base_xy.length, line = (a, b, style, dash) => {
    g.strokeStyle = style; g.lineWidth = 3; g.setLineDash(dash); g.beginPath();
    let pen = false;
    for (let i = a; i <= b; i++) {
      const q = proj(cam, [o.base_xy[i][0], o.base_xy[i][1], 0.0], W, H);
      if (!q) { pen = false; continue; }
      pen ? g.lineTo(ox + q[0], oy + q[1]) : g.moveTo(ox + q[0], oy + q[1]); pen = true;
    }
    g.stroke(); g.setLineDash([]);
  };
  line(0, step, "rgba(255,214,10,.95)", []);
  line(step, n - 1, "rgba(255,214,10,.4)", [6, 5]);
}

function drawSkeleton(g, pts, hover, trunk) {
  const legGroup = groupLeg(), focus = ovFocus || hover;
  g.lineCap = "round"; g.lineJoin = "round";
  if (trunk) {   // the trunk as a quad through the four hips
    const q = ["FR", "FL", "RL", "RR"].map((l) => pts[`${l}_hip`]);
    if (q.every(Boolean)) { g.fillStyle = "rgba(160,175,200,.22)"; g.strokeStyle = "rgba(200,212,230,.7)"; g.lineWidth = 2; g.beginPath(); q.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath(); g.fill(); g.stroke(); }
  }
  for (const leg of Object.keys(LEG_COLOR)) {
    g.globalAlpha = legGroup && legGroup !== leg ? 0.3 : 1;
    g.strokeStyle = LEG_COLOR[leg]; g.lineWidth = 3;
    const chain = [pts.base, pts[`${leg}_hip`], pts[`${leg}_thigh`], pts[`${leg}_calf`], pts[`${leg}_foot`]];
    g.beginPath(); chain.forEach((q, i) => { if (!q) return; i === 0 || !chain[i - 1] ? g.moveTo(q[0], q[1]) : g.lineTo(q[0], q[1]); }); g.stroke();
    for (const part of ["hip", "thigh", "calf", "foot"]) {
      const q = pts[`${leg}_${part}`]; if (!q) continue;
      const name = `${leg}_${part}`, hot = focus === name;
      g.fillStyle = part === "foot" ? "#fff" : LEG_COLOR[leg]; g.strokeStyle = part === "foot" ? LEG_COLOR[leg] : "#fff"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(q[0], q[1], hot ? 8 : (part === "calf" ? 5.5 : 4.5), 0, 7); g.fill(); g.stroke();
      if (hot) { g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.arc(q[0], q[1], 12, 0, 7); g.stroke(); }
    }
  }
  g.globalAlpha = 1;
  if (focus && pts[focus]) {
    const q = pts[focus], text = JOINT_JP(focus);
    g.font = "12px system-ui, sans-serif"; g.textAlign = "left"; const w = g.measureText(text).width + 12;
    g.fillStyle = "rgba(15,18,25,.85)"; g.fillRect(q[0] + 12, q[1] - 24, w, 20); g.fillStyle = "#fff"; g.textBaseline = "middle"; g.fillText(text, q[0] + 18, q[1] - 14);
  }
}

// A top-down map of the whole path with the robot's current position: x to the right, y up (as the simulator's axes).
function drawMinimap(g, o, step) {
  const xy = o.base_xy, size = 140, x0 = 10, y0 = 10, [sx, sy] = xy[0];
  const xs = xy.map((p) => p[0] - sx), ys = xy.map((p) => p[1] - sy);
  const lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 0.5), pad = 18, sc = (size - 2 * pad) / span;
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
  const P = (p) => [x0 + size / 2 + (p[0] - sx - cx) * sc, y0 + size / 2 - (p[1] - sy - cy) * sc];
  g.fillStyle = "rgba(15,18,25,.74)"; g.strokeStyle = "rgba(255,255,255,.25)"; g.lineWidth = 1;
  g.beginPath(); g.rect(x0, y0, size, size); g.fill(); g.stroke();
  g.font = "9.5px system-ui, sans-serif"; g.fillStyle = "rgba(235,240,250,.8)";
  if (ovScale) {   // grid with values (origin = the start position)
    const st = nice(span); g.strokeStyle = "rgba(200,215,235,.2)"; g.textBaseline = "top"; g.textAlign = "center";
    for (let v = Math.ceil((lo[0] + cx - span / 2) / st) * st; v <= cx + span / 2; v += st) { const q = P([sx + v, sy]); if (q[0] > x0 + 3 && q[0] < x0 + size - 3) { g.beginPath(); g.moveTo(q[0], y0 + 3); g.lineTo(q[0], y0 + size - 12); g.stroke(); g.fillText(+v.toFixed(2) + "", q[0], y0 + size - 11); } }
    g.textAlign = "left"; g.textBaseline = "middle";
    for (let v = Math.ceil((cy - span / 2) / st) * st; v <= cy + span / 2; v += st) { const q = P([sx, sy + v]); if (q[1] > y0 + 6 && q[1] < y0 + size - 14) { g.beginPath(); g.moveTo(x0 + 3, q[1]); g.lineTo(x0 + size - 3, q[1]); g.stroke(); g.fillText(+v.toFixed(2) + "", x0 + 3, q[1] - 5); } }
  }
  g.strokeStyle = "rgba(255,214,10,.4)"; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath();
  xy.forEach((p, i) => { const q = P(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); }); g.stroke(); g.setLineDash([]);
  g.strokeStyle = "rgba(255,214,10,.95)"; g.lineWidth = 2.5; g.beginPath();
  xy.slice(0, step + 1).forEach((p, i) => { const q = P(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); }); g.stroke();
  const s0 = P(xy[0]), c = P(xy[Math.min(step, xy.length - 1)]);
  g.fillStyle = "#fff"; g.beginPath(); g.arc(s0[0], s0[1], 3, 0, 7); g.fill();
  g.fillStyle = "#ffd60a"; g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc(c[0], c[1], 5, 0, 7); g.fill(); g.stroke();
  // axis arrows (x right, y up) in the corner of the map
  const ax = x0 + size - 26, ay = y0 + 24; g.lineWidth = 1.8; g.font = "bold 10px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.strokeStyle = g.fillStyle = AXIS_COLOR.x; g.beginPath(); g.moveTo(ax - 12, ay); g.lineTo(ax + 8, ay); g.stroke(); g.fillText("x", ax + 14, ay);
  g.strokeStyle = g.fillStyle = AXIS_COLOR.y; g.beginPath(); g.moveTo(ax - 12, ay); g.lineTo(ax - 12, ay - 18); g.stroke(); g.fillText("y", ax - 12, ay - 24);
  g.fillStyle = "rgba(255,255,255,.8)"; g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.textBaseline = "top";
  g.fillText(ovScale ? "上から見た軌跡（m）" : "上から見た軌跡", x0 + 4, y0 + size + 3);
}

function groupLeg() { const sel = $("tracegroup"); const g = trace && trace.groups && trace.groups[Number(sel.value) || 0]; return g && g.leg || null; }
const JOINT_JP = (n) => { const [leg, part] = n.split("_"); return n === "base" ? "胴体" : `${LEGS[leg]}の${{hip: "付け根（hip）", thigh: "もも（thigh）", calf: "ひざ（calf）", foot: "足先"}[part]}`; };
let ovMouse = null, ovHover = null;

function drawOverlay() {
  const cv = $("overlay"), v = $("video");
  const dpr = window.devicePixelRatio || 1, bw = cv.clientWidth, bh = cv.clientHeight;
  if (cv.width !== Math.round(bw * dpr) || cv.height !== Math.round(bh * dpr)) { cv.width = Math.round(bw * dpr); cv.height = Math.round(bh * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, bw, bh);
  if (!ov || !ov.pts || !ov.pts.length || v.style.display === "none") return;
  const free = viewFree;
  if (!free && !ovOn && !ovScale) return;
  const step = Math.max(0, Math.min(ov.pts.length - 1, Math.round(v.currentTime / ov.dt)));
  let cam, W, H, ox = 0, oy = 0;
  if (free) {
    const ch = $("freectl").offsetHeight || 0;
    W = bw; H = bh - ch; g.fillStyle = "#0a141c"; g.fillRect(0, 0, W, H);
    cam = freeCam(ov, step, W / H);
    const t = Math.min(ov.pts.length - 1, step); $("fslider").max = ov.pts.length - 1; if (!fdrag || !fdrag.slider) $("fslider").value = t;
    $("ftime").textContent = `${(t * ov.dt).toFixed(1)} / ${((ov.pts.length - 1) * ov.dt).toFixed(1)} 秒`;
    $("fplay").textContent = v.paused ? "▶" : "❚❚";
  } else {
    const [rw, rh] = ov.res, s = Math.min(bw / rw, bh / rh);
    W = rw * s; H = rh * s; ox = (bw - W) / 2; oy = (bh - H) / 2; cam = recordedCam(ov, step);
  }
  const clip = () => { g.save(); g.beginPath(); g.rect(ox, oy, W, H); g.clip(); };
  clip();
  if (free || ovScale) drawGrid(g, cam, ov, step, W, H, ox, oy, ovScale);
  if (free || ovOn) {
    drawTrail(g, ov, step, cam, W, H, ox, oy);
    const pts = skeletonPoints(ov, step, cam, W, H, ox, oy);
    let hover = null, bd = 18;
    if (ovMouse) for (const [n, q] of Object.entries(pts)) { const d = Math.hypot(q[0] - ovMouse[0], q[1] - ovMouse[1]); if (d < bd) { bd = d; hover = n; } }
    ovHover = hover;
    drawSkeleton(g, pts, hover, free);
  }
  g.restore();
  if (free || ovOn || ovScale) { drawMinimap(g, ov, step); drawAxes(g, cam, bw - 52, (free ? H : bh) - 74); }
}
(function loop() { drawOverlay(); requestAnimationFrame(loop); })();

const criterionLabel = (metric) => ((S.criteria || []).find((c) => c.metric === metric) || {}).comment || metric;

// Empty every panel (a new experiment is opened: nothing of the previous one may stay on screen).
function clearView(msg) {
  for (const id of ["loopbadges", "flow", "scorerow", "fixblock", "chgblock", "tracelegend", "tracegroup"]) $(id).textContent = "";
  $("story").hidden = true; $("retarrow").textContent = ""; $("progress").style.width = "0%";
  $("instrline").textContent = ""; $("taskline").textContent = ""; $("resultcap").textContent = "";
  const v = $("video"); v.pause(); v.removeAttribute("src"); v.style.display = "none"; shownVideo = "";
  $("novideo").textContent = msg || "学習が終わると、ここに結果が表示されます"; $("novideo").style.display = "flex";
  $("ovbtn").hidden = $("scalebtn").hidden = $("viewbtn").hidden = true; $("freectl").hidden = true; ov = null; ovKey = "";
  $("loopcard").hidden = true;
  drawTrace(null);
}

function render() {
  if (!S || !exp) return;   // nothing is shown while no experiment is chosen ("（選んでください）")
  $("loopcard").hidden = false;
  const vt = virtualNow();
  const loops = viewAt(S, vt);
  const finished = S.status === "done" && (!replay || vt >= S.t_end);
  const badge = $("modebadge");
  $("replaybar").hidden = !replay; $("replaybtn").hidden = !!replay;
  if (replay) { badge.textContent = `経過を再現中（×${replay.speed}）` + (S.synthetic ? " ・時刻は記録から再構成" : ""); badge.className = "badge replay"; }
  else if (S.status === "running") { badge.textContent = "ライブ"; badge.className = "badge live"; }
  else { badge.textContent = S.status === "stalled" ? "停止？（更新なし）" : S.status === "failed" ? "異常終了" : S.status === "stopped" ? "強制終了しました" : "完了"; badge.className = "badge"; }
  $("stopbtn").hidden = !(S.can_stop && !replay);

  $("instrline").textContent = S.instruction ? `「${S.instruction}」` : "";
  const ti = S.task_info;
  $("taskline").textContent = (ti ? `芸: ${ti.name} ／ 成功の基準 ${ti.criteria} 項目（人が承認済み）` : (S.task ? `成功条件: ${S.task}` : ""))
    + (S.hint_info ? ` ／ 設計ヒント: ${S.hint_info.name}（人が書いた助言）` : "") + (S.prev_exp ? ` ／ 前の実験 ${S.prev_exp} の結果を引き継ぎ` : "");
  const {cur, best, flow: fl, result} = pickLoops(loops, finished);
  $("loopline").textContent = S.max_loops ? `学習のループ（最大 ${S.max_loops} 周。番号をクリックすると、その周を表示）` : "学習のループ";

  const bl = finished ? bestLoop(loops) : null;
  $("viewing").textContent = fl ? `表示中: ${fl.loop}周目${S.max_loops ? " / " + S.max_loops : ""}${bl && bl.loop === fl.loop ? " ★" : ""}` : "";
  $("tracelabel").textContent = `動きのグラフ${result ? `（${result.loop}周目）` : ""} — 動画の再生位置と同期。クリックでその場面へ、ドラッグで拡大`;
  const safe = (f) => { try { f(); } catch (e) { console.error(e); } };   // one broken part must not stop the rest of the screen
  safe(() => renderBadges(loops, fl, bl));
  safe(() => renderFlow(fl, vt, finished));
  safe(() => renderScore(fl));
  safe(() => renderStory(fl, loops));
  safe(() => renderNow(cur, vt, finished));
  safe(() => renderImprove(loops, finished));

  const v = $("video");
  if (result && result.video) {
    ensureOverlay(result);
    const vid = videoOverride[`${exp}/${result.loop}`] || result.video;
    $("novideo").style.display = "none"; v.style.display = "block";
    if (shownVideo !== vid) {
      shownVideo = vid; v.src = videoUrl(vid); v.play().catch(() => {});
      const sk = new URLSearchParams(location.search).get("seek");   // ?seek=SECONDS: stop the video there (screenshots)
      if (sk !== null) v.onloadedmetadata = () => { v.currentTime = Number(sk); v.pause(); };
    }
  } else { v.style.display = "none"; $("novideo").style.display = "flex"; shownVideo = ""; }
  // while a newer loop is still running, say that the picture is the previous loop's result
  $("resultcap").textContent = result && cur && cur.loop > result.loop && !done(cur, "record") && selLoop == null
    ? `表示中: ${result.loop}周目の結果（${cur.loop}周目は学習中）`
    : (result ? `${result.loop}周目の結果` + (best && best.loop === result.loop && selLoop == null ? "（★ 最も評価の高い周）" : "") : "");
  drawTrace(result).catch((e) => console.error(e));
}

// Outcome colour: "good" (green), "mid" (orange) or "bad" (red).
// Green always means "passed" (the run goes on after anything that is not green). Orange = close, red = far.
const qNumeric = (n) => n.passed ? "good" : n.n_passed / n.n_criteria >= 0.7 ? "mid" : "bad";
const qVlm = (e) => (e.success && e.score >= 8) ? "good" : e.score >= 5 ? "mid" : "bad";
function qLoop(lp) {
  if (lp.decision && done(lp, "decision") && lp.decision.success) return "good";
  if (!(lp.numeric && lp.evaluation && done(lp, "vlm"))) return "";
  const v = (lp.numeric.n_passed / lp.numeric.n_criteria + lp.evaluation.score / 10) / 2;
  return v >= 0.55 ? "mid" : "bad";
}
// The numeric check missed only a little (a tenth of the items at most) while the AI scorer passed it: "あと一歩".
function nearMiss(lp) {
  const n = lp.numeric, e = lp.evaluation;
  return !!(n && e && !n.passed && n.n_passed / n.n_criteria >= 0.9 && e.success && e.score >= 8);
}
function judgeNote(lp) {
  const d = lp.decision && lp.decision.disagreement;
  if (!d) return "";
  if (d === "numeric_pass_vlm_fail") return "数値判定は全項目合格でしたが、AI採点は不合格でした（基準が緩い可能性があります）。";
  const miss = (lp.numeric.criteria || []).filter((c) => !c.passed).map((c) => criterionLabel(c.metric));
  return nearMiss(lp)
    ? `あと一歩: AI採点は合格で、数値判定は ${miss.length} 項目（${miss.join("、")}）だけ基準に届きませんでした。成功条件はすべての項目を満たす必要があるため、この周は成功とは数えません。`
    : "数値判定は不合格でしたが、AI採点は合格でした（基準が厳しすぎる可能性があります）。";
}

function badgeInfo(lp) {
  const running = STAGES.find((k) => lp.stages[k] && lp.stages[k].status === "running");
  if (done(lp, "decision") && lp.decision) return {mark: lp.decision.success ? "✓" : "✗", sub: lp.evaluation ? `${lp.evaluation.score}/10` : ""};
  if (done(lp, "vlm") && lp.evaluation) return {mark: lp.evaluation.success ? "✓" : "✗", sub: `${lp.evaluation.score}/10`};
  return {mark: "", sub: running ? LABELS[running] : "…"};
}

function renderBadges(loops, fl, best) {
  const finished0 = !!best;
  const box = $("loopbadges");
  box.textContent = "";
  for (const lp of loops) {
    const info = badgeInfo(lp);
    const q = qLoop(lp), running = !info.mark && info.sub !== "…" && !done(lp, "decision");
    const b = el("span", `${lp.loop}${info.mark ? " " + info.mark : ""}${best && best.loop === lp.loop ? " ★" : ""}`, "lb" + (fl && fl.loop === lp.loop ? " sel" : "") + (q ? " q-" + q : "") + (running && !finished0 ? " run-badge" : ""));
    b.appendChild(el("small", info.sub));
    b.onclick = () => { selLoop = selLoop === lp.loop ? null : lp.loop; shownVideo = ""; render(); };
    box.appendChild(b);
  }
}

function renderFlow(lp, vt, finished) {
  const flow = $("flow");
  flow.textContent = "";
  for (const k of STAGES) {
    let cls = "", sub = "";
    if (k === "instruction") cls = S.instruction ? "done" : "";
    else if (lp && lp.stages[k]) {
      const st = lp.stages[k]; cls = st.status;
      const secs = st.end != null ? `${(st.end - st.start).toFixed(0)}秒` : "";
      if (k === "reward" || k === "check" || k === "record") sub = secs;
      if (k === "train") sub = st.status === "running" ? `${Math.round(trainFrac(lp, vt) * (lp.train && lp.train.total || 0))} / ${(lp.train && lp.train.total) || "?"}` : secs;
      if (k === "numeric" && st.status === "done" && lp.numeric) sub = `${lp.numeric.n_passed}/${lp.numeric.n_criteria}`;
      if (k === "vlm" && st.status === "done" && lp.evaluation) sub = `${lp.evaluation.score}/10`;
      if (k === "decision" && st.status === "done" && lp.decision) sub = lp.decision.success ? "成功" : nearMiss(lp) ? "あと一歩" : "修正へ";
    }
    let q = "";
    if (lp && k === "numeric" && done(lp, "numeric") && lp.numeric) q = qNumeric(lp.numeric);
    if (lp && k === "vlm" && done(lp, "vlm") && lp.evaluation) q = qVlm(lp.evaluation);
    if (lp && k === "decision" && done(lp, "decision")) q = qLoop(lp);
    const li = el("li", LABELS[k], cls + (q ? " q-" + q : ""));
    if (sub) li.appendChild(el("small", sub));
    li.title = "クリックで、この段階の中身（実装コード）を見る";
    li.onclick = () => openStage(k, lp);
    flow.appendChild(li);
  }
  const f = lp && lp.stages.train && !done(lp, "train") ? trainFrac(lp, vt) : (finished ? 1 : 0);
  $("progress").style.width = `${(f * 100).toFixed(0)}%`;
  $("retarrow").textContent = finished && S.stop_reason === "success" ? "数値判定とAI採点の両方が合格 → 終了"
    : "どちらかが不合格 → AIの指摘を渡して次の周へ" + (S.max_loops ? `（上限 ${S.max_loops} 周）` : "");
}

// After a run that did not succeed (or whose two judgements disagreed): offer to revise the success spec and retrain.
function renderImprove(loops, finished) {
  const box = $("improve");
  const dis = loops.some((l) => l.decision && l.decision.disagreement);
  const ok = S.stop_reason === "success";
  const show = canRun && finished && !replay && S.status !== "running";   // after a success too: the AI's remarks can be taken into a new run
  box.hidden = !show; box.textContent = "";
  if (!show) return;
  box.appendChild(el("div", ok ? "成功しました。AI採点の指摘（改善点）を受けて、基準を厳しくして、さらに良い動きを目指す再学習もできます。"
    : dis ? "数値判定とAI採点の結果が分かれた周があります。数値判定の基準が緩すぎる、または厳しすぎる可能性があります。" : "成功条件を満たさずに終わりました。基準を見直して、もう一度学習できます。", "muted"));
  const b = el("button", ok ? "AIの指摘を受けて再学習…" : "成功条件を見直して再学習…", "ghost"); b.onclick = () => improveFlow();
  box.appendChild(b);
}

const fmtDur = (s) => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}分${String(s % 60).padStart(2, "0")}秒`; };
// "Something is happening": what is running now, for how long, and (training) how far it is, with the reward curve.
function renderNow(cur, vt, finished) {
  const box = $("nowline"), sp = $("spark");
  const run = cur && !finished && (S.status === "running" || S.status === "stalled" || replay) && STAGES.find((k) => cur.stages[k] && cur.stages[k].status === "running");
  $("progress").classList.toggle("busy", !!run);
  if (!run) { box.hidden = true; sp.hidden = true; document.title = "Genesis Dojo"; return; }
  const st = cur.stages[run], el_ = vt - st.start;
  let text = `${LABELS[run]}を実行中 ・ 経過 ${fmtDur(el_)}`;
  if (run === "train") {
    const f = trainFrac(cur, vt), total = (cur.train && cur.train.total) || 0;
    const mean = cur.train && cur.train.curve && cur.train.curve.length ? cur.train.curve[cur.train.curve.length - 1][1] : null;
    text += ` ・ ${Math.round(f * total)} / ${total || "?"} 反復`;
    if (f > 0.03) text += ` ・ 残り約 ${fmtDur(el_ / f - el_)}`;
    if (mean != null && !replay) text += ` ・ 平均報酬 ${Number(mean).toFixed(1)}`;
    drawSpark(sp, cur, f);
  } else sp.hidden = true;
  box.textContent = ""; box.appendChild(el("span", null, "dot")); box.appendChild(document.createTextNode(text)); box.hidden = false;
  document.title = `● ${LABELS[run]}中 ${cur.loop}周目 ― Genesis Dojo`;
}
function drawSpark(cv, lp, f) {
  const pts = lp.train && lp.train.curve ? lp.train.curve : [], total = lp.train && lp.train.total || 300;
  const shown = replay ? pts.filter((p) => p[0] <= f * total) : pts;
  if (shown.length < 2) { cv.hidden = true; return; }
  cv.hidden = false;
  const css = getComputedStyle(document.body), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const ys = pts.map((p) => p[1]), lo = Math.min(...ys), hi = Math.max(...ys);
  const X = (i) => 4 + (i / total) * (w - 8), Y = (y) => h - 14 - ((y - lo) / ((hi - lo) || 1)) * (h - 22);
  g.strokeStyle = css.getPropertyValue("--accent"); g.lineWidth = 2; g.beginPath();
  shown.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.stroke();
  const last = shown[shown.length - 1]; g.fillStyle = css.getPropertyValue("--accent"); g.beginPath(); g.arc(X(last[0]), Y(last[1]), 3.5, 0, 7); g.fill();
  g.fillStyle = css.getPropertyValue("--mute"); g.font = "10px system-ui, sans-serif"; g.fillText("平均報酬の推移（右上がり = 学習が進んでいる）", 4, h - 2);
}

function renderScore(lp) {
  const box = $("scorerow");
  box.textContent = "";
  if (!lp) return;
  if (lp.stages.numeric && done(lp, "numeric") && lp.numeric && lp.numeric.criteria) {
    const n = lp.numeric, row = el("div", null, "sc");
    const head = el("div"); head.appendChild(el("b", "数値判定")); head.appendChild(el("span", ` ${n.n_passed} / ${n.n_criteria} 項目が合格`, "val" + (n.passed ? "" : " bad")));
    const dots = el("span", null, "dots");
    for (const c of n.criteria) { const i = el("i", null, c.passed ? "" : "x"); i.title = `${criterionLabel(c.metric)}（基準: ${c.rule}）`; dots.appendChild(i); }
    head.appendChild(dots);
    row.appendChild(head);
    const bad = n.criteria.filter((c) => !c.passed).map((c) => criterionLabel(c.metric));
    row.appendChild(el("div", bad.length ? "不合格: " + bad.join("、") : "動画から測った数値が、人が決めた基準をすべて満たした", "muted"));
    if (lp.numeric_label === "retro") row.appendChild(el("div", "後から録画し直して判定", "muted"));
    box.appendChild(row);
  }
  if (done(lp, "vlm") && lp.evaluation) {
    const e = lp.evaluation, row = el("div", null, "sc");
    const head = el("div"); head.appendChild(el("b", "AI採点")); head.appendChild(el("span", ` ${e.score} / 10 点`, "val" + (e.success && e.score >= 8 ? "" : " bad")));
    head.appendChild(el("span", "（8点以上で合格）", "muted"));
    row.appendChild(head);
    const text = e.observations || e.reasoning || e.comment || e.summary || "";
    if (text) { row.appendChild(el("div", text, "muted comment")); }
    box.appendChild(row);
  }
  if (lp.decision && lp.decision.disagreement && done(lp, "decision")) box.appendChild(el("div", judgeNote(lp), "muted"));
  if (lp.error) box.appendChild(el("div", "エラー: " + lp.error, "bad"));
}

// The two texts under the flow, in the order they happen: the AI score of the previous loop (its advice) is handed to
// "報酬生成" of this loop, which writes the new reward. Hovering a block marks that node in the flow.
function renderStory(lp, loops) {
  const card = $("story"), fix = $("fixblock"), chg = $("chgblock");
  fix.textContent = ""; chg.textContent = "";
  const prev = lp && loops.find((l) => l.loop === lp.loop - 1);
  const advice = prev && done(prev, "vlm") && prev.evaluation && (prev.evaluation.next_fix || "");
  const summary = lp && done(lp, "reward") && lp.reward_summary;
  const head = (stageLabel, who, why) => {
    const h = el("div", null, "chips");
    h.appendChild(el("span", stageLabel, "chip2")); h.appendChild(el("span", who)); h.appendChild(el("span", "— " + why, "why"));
    return h;
  };
  if (advice) { fix.appendChild(head("AI採点", `${prev.loop}周目`, `不合格だったので、次の周の「報酬生成」へ渡された指摘`)); fix.appendChild(el("span", advice)); }
  if (summary) { chg.appendChild(head("報酬生成", `${lp.loop}周目`, lp.loop === 1 ? "指示から、LLMが最初に書いた報酬の要点" : "指摘を受けて、LLMが書き換えた報酬の要点")); chg.appendChild(el("span", summary)); }
  fix.hidden = !advice; chg.hidden = !summary;
  $("storyarrow").hidden = !(advice && summary);
  card.hidden = !advice && !summary;
}
const hlStage = (on) => { const i = STAGES.indexOf("reward"); const li = $("flow").children[i]; if (li) li.classList.toggle("hl", on); };

// ---- stage drawer: what a stage does, this loop's data, the command and the implementation ----
function section(title, open) { const d = el("details"); d.open = !!open; d.appendChild(el("summary", title)); return d; }
const pre = (text) => el("pre", text);

async function openStage(stage, lp) {
  const [title, what] = STAGE_DOC[stage];
  $("dtitle").textContent = title;
  const body = $("dbody"); body.textContent = "";
  body.appendChild(el("p", what));
  const data = section(lp ? `${lp.loop}周目のデータ` : "この周のデータ", true);
  stageData(stage, lp, data);
  body.appendChild(data);
  const cmd = stageCommand(stage, lp);
  if (cmd) { const d = section("実行コマンド（リポジトリのルートから）", true); d.appendChild(pre(cmd)); body.appendChild(d); }
  const src = section("実装コード", true);
  src.appendChild(el("p", "読み込み中…", "muted"));
  body.appendChild(src);
  $("drawer").hidden = false; $("backdrop").hidden = false;
  try {
    const files = await api(`/api/source?stage=${stage}&exp=${encodeURIComponent(exp || "")}`);
    src.textContent = ""; src.appendChild(el("summary", "実装コード"));
    for (const f of files) { const d = section(f.path, files.length === 1); d.appendChild(pre(f.text)); src.appendChild(d); }
    if (!files.length) src.appendChild(el("p", "表示できるコードがありません", "muted"));
  } catch (e) { src.textContent = "コードを読み込めませんでした"; }
}

function stageCommand(stage, lp) {
  const name = lp ? lp.name : "<実験名>_L1", task = S && S.task;
  if (stage === "instruction") return `python scripts/loop.py --exp ${S ? S.exp : "<実験名>"} --instruction "${S ? S.instruction : "<指示>"}" \\\n  --success-spec configs/tasks/${task || "<芸>"}.success.yaml --base-config configs/base.yaml`;
  if (stage === "train") return `python scripts/train.py configs/${name}.yaml`;
  if (stage === "record") return `python scripts/record.py ${name}`;
  if (stage === "numeric") return `python scripts/check_success.py ${name} --spec configs/tasks/${task || "<芸>"}.success.yaml`;
  return "";
}

function stageData(stage, lp, box) {
  const add = (...n) => n.forEach((x) => box.appendChild(x));
  if (!lp && stage !== "instruction") { add(el("p", "まだデータがありません", "muted")); return; }
  if (stage === "instruction") {
    add(el("p", S && S.instruction ? `指示: 「${S.instruction}」` : ""));
    if (S && S.task_info) add(el("p", `選ばれた芸: ${S.task_info.name}（${S.task_info.description}）`));
    const t = el("table");
    for (const c of (S && S.criteria) || []) { const tr = el("tr"); tr.appendChild(el("td", c.comment || c.metric)); tr.appendChild(el("td", c.rule, "muted")); t.appendChild(tr); }
    add(el("p", "成功の基準（人が事前に承認して固定）", "muted"), t);
  } else if (stage === "reward") {
    if (lp.reward_summary) add(el("p", lp.reward_summary));
    if (lp.reward_code) add(el("p", "LLMが書いた報酬関数（この周）", "muted"), pre(lp.reward_code));
    else add(el("p", "報酬コードを生成中…", "muted"));
  } else if (stage === "check") {
    for (const a of lp.check_attempts || []) add(el("p", `試行 ${a.attempt}: ${a.ok ? "合格" : "不合格 — " + (a.message || "")}`));
    if (!(lp.check_attempts || []).length) add(el("p", "検証の記録がありません", "muted"));
  } else if (stage === "train") {
    const t = lp.train;
    if (!t) { add(el("p", "学習中…", "muted")); return; }
    add(el("p", `学習 ${t.iteration ?? t.iterations ?? "?"} / ${t.total || "?"} 反復、${t.seconds ? Math.round(t.seconds) + "秒" : ""}、最終の平均報酬 ${t.final_mean_reward ?? "—"}`));
    const cv = el("canvas"); cv.width = 640; cv.height = 200; add(el("p", "平均報酬の推移（横: 反復、縦: 平均報酬）", "muted"), cv);
    requestAnimationFrame(() => drawCurve(cv, t));
  } else if (stage === "record") {
    add(el("p", lp.video ? `動画: ${lp.video}` : "録画中…"));
  } else if (stage === "numeric") {
    const t = el("table");
    for (const c of (lp.numeric && lp.numeric.criteria) || []) {
      const tr = el("tr"); tr.appendChild(el("td", criterionLabel(c.metric))); tr.appendChild(el("td", `基準 ${c.rule}`, "muted"));
      tr.appendChild(el("td", `${c.measured == null ? "—" : c.measured} ${c.passed ? "✓" : "✗"}`, c.passed ? "" : "bad")); t.appendChild(tr);
    }
    add(t);
  } else if (stage === "vlm") {
    const e = lp.evaluation;
    if (!e) { add(el("p", "採点中…", "muted")); return; }
    add(el("p", `${e.score} / 10 点（${e.success ? "成功" : "失敗"}と判定）`), el("p", e.observations || ""));
    if (Array.isArray(e.problems) && e.problems.length) { const ul = el("ul"); e.problems.forEach((x) => ul.appendChild(el("li", String(x)))); add(el("p", "問題点", "muted"), ul); }
    if (e.next_fix) add(el("p", "次の修正案", "muted"), el("p", e.next_fix));
  } else if (stage === "decision") {
    const d = lp.decision;
    if (!d) { add(el("p", "まだ判定されていません", "muted")); return; }
    add(el("p", `数値判定: ${d.numeric_pass ? "合格" : "不合格"} ／ AI採点: ${d.vlm_pass ? "合格" : "不合格"} → ${d.success ? "成功（終了）" : "次の周へ"}`));
    if (d.disagreement) add(el("p", judgeNote(lp), "muted"));
  }
}

function drawCurve(cv, t) {
  const g = cv.getContext("2d"), css = getComputedStyle(document.body);
  const pts = t.curve || [];
  if (pts.length < 2) return;
  const ys = pts.map((p) => p[1]), lo = Math.min(...ys), hi = Math.max(...ys), pad = 34, W = cv.width, H = cv.height, total = t.total || pts.length;
  const X = (i) => pad + (i / total) * (W - pad - 10), Y = (y) => H - 22 - ((y - lo) / ((hi - lo) || 1)) * (H - 22 - 10);
  g.strokeStyle = css.getPropertyValue("--line"); g.fillStyle = css.getPropertyValue("--mute"); g.font = "11px sans-serif";
  g.beginPath(); g.moveTo(pad, 10); g.lineTo(pad, H - 22); g.lineTo(W - 10, H - 22); g.stroke();
  g.fillText(hi.toFixed(1), 2, 16); g.fillText(lo.toFixed(1), 2, H - 22);
  g.strokeStyle = css.getPropertyValue("--accent"); g.lineWidth = 2; g.beginPath();
  pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.stroke();
}

function closeDrawer() { $("drawer").hidden = true; $("modal").hidden = true; $("backdrop").hidden = true; }

// ---- trace: time series of the recorded run (joint angles, foot / head / knee heights, posture) ----
// Hover: crosshair + values of every line (the nearest line is highlighted). Click: seek the video. Drag: zoom; double-click: reset.
// Legend: click to hide / show a line. Dashed horizontal lines are the thresholds of the success spec.
const LEGS = {FR: "右前脚", FL: "左前脚", RR: "右後脚", RL: "左後脚"};
const COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7"];
let trace = null, traceKey = "";   // trace of the loop on screen (null = none), cached by "exp/loop"
let lastTraceLoop = null;
const traceFailAt = {};   // "exp/loop" -> when loading its data last failed
const tv = {range: null, hover: null, drag: null, hidden: new Set()};   // view state of the chart

function traceGroups(t) {
  const s = t.series, g = [];
  const pick = (title, defs, unit, refs) => {
    const lines = defs.filter(([, k]) => s[k]).map(([label, k]) => ({label, data: s[k].map((v) => (v == null ? NaN : v))}));   // null = a value the simulation could not give
    if (lines.length) g.push({title, unit, lines, refs: refs || []});
  };
  pick("前足の高さ", [["左前足", "foot_z_FL"], ["右前足", "foot_z_FR"]], "m", [{y: 0.06, label: "持ち上げの基準 0.06"}]);
  pick("後ろ足の高さ", [["左後足", "foot_z_RL"], ["右後足", "foot_z_RR"]], "m");
  pick("頭と前膝の高さ", [["頭", "head_z"], ["左前膝", "knee_z_FL"], ["右前膝", "knee_z_FR"]], "m", [{y: 0.2, label: "頭の基準 0.2"}, {y: 0.12, label: "膝の基準 0.12"}]);
  pick("胴体の傾き（前後・左右）", [["前後の傾き（頭が下がると負）", "nose_up_deg"], ["左右の傾き（横に倒れると±）", "roll_deg"]], "度");
  pick("胴体の高さと進んだ距離", [["胴体の高さ", "height_m"], ["開始位置からの距離", "xy_dist_m"]], "m");
  if (t.joint_names) {
    for (const leg of Object.keys(LEGS)) {
      const idx = t.joint_names.map((n, i) => [n, i]).filter(([n]) => n.startsWith(leg + "_"));
      pick(`関節の角度: ${LEGS[leg]}`, idx.map(([n, i]) => [n.slice(leg.length + 1).replace("_joint", ""), "q_" + i]), "rad");
      const last = g[g.length - 1];
      if (last && last.title.startsWith("関節")) { last.leg = leg; last.lines.forEach((l) => { l.joint = `${leg}_${l.label}`; }); }
    }
  }
  return g;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * p).find((x) => x >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return {ticks: out, step};
}

const chartGeom = (cv, grp, n) => {
  const padL = 52, padR = 12, padT = 10, padB = 26, W = cv.clientWidth, H = cv.clientHeight;
  const [i0, i1] = tv.range || [0, n - 1];
  const vis = grp.lines.filter((l) => !tv.hidden.has(l.label));
  const vals = (vis.length ? vis : grp.lines).flatMap((l) => l.data.slice(i0, i1 + 1)).concat(grp.refs.map((r) => r.y)).filter(Number.isFinite);
  let lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
  const m = (hi - lo || 1) * 0.06; lo -= m; hi += m;
  return {padL, padR, padT, padB, W, H, i0, i1, lo, hi,
          X: (i) => padL + ((i - i0) / Math.max(1, i1 - i0)) * (W - padL - padR),
          Y: (y) => H - padB - ((y - lo) / (hi - lo)) * (H - padB - padT),
          iAt: (px) => Math.max(i0, Math.min(i1, Math.round(i0 + ((px - padL) / (W - padL - padR)) * (i1 - i0))))};
};

function paintTrace() {
  const cv = $("trace"), grp = trace && trace.groups && trace.groups[Number($("tracegroup").value) || 0];
  if (!grp) return;
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.body), line = css.getPropertyValue("--line").trim(), mute = css.getPropertyValue("--mute").trim(), ink = css.getPropertyValue("--ink").trim();
  const n = Math.max(...grp.lines.map((l) => l.data.length)), G = chartGeom(cv, grp, n), dt = trace.dt;
  g.font = "11px system-ui, sans-serif";
  // grid + y ticks
  const yt = niceTicks(G.lo, G.hi, 5);
  g.lineWidth = 1; g.textAlign = "right"; g.textBaseline = "middle";
  for (const v of yt.ticks) {
    const y = Math.round(G.Y(v)) + 0.5;
    g.strokeStyle = line; g.beginPath(); g.moveTo(G.padL, y); g.lineTo(w - G.padR, y); g.stroke();
    g.fillStyle = mute; g.fillText(+v.toFixed(3) + "", G.padL - 6, y);
  }
  g.save(); g.translate(11, G.padT + (h - G.padT - G.padB) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillStyle = mute; g.fillText(grp.unit ? `[${grp.unit}]` : "", 0, 0); g.restore();
  // x ticks (seconds)
  const xt = niceTicks(G.i0 * dt, G.i1 * dt, 6);
  g.textAlign = "center"; g.textBaseline = "top";
  for (const t of xt.ticks) {
    const x = Math.round(G.X(t / dt)) + 0.5;
    g.strokeStyle = line; g.beginPath(); g.moveTo(x, h - G.padB); g.lineTo(x, h - G.padB + 4); g.stroke();
    g.fillStyle = mute; g.fillText(+t.toFixed(2) + "秒", x, h - G.padB + 7);
  }
  g.strokeStyle = mute; g.beginPath(); g.moveTo(G.padL + 0.5, G.padT); g.lineTo(G.padL + 0.5, h - G.padB + 0.5); g.lineTo(w - G.padR, h - G.padB + 0.5); g.stroke();
  // threshold lines
  g.save(); g.beginPath(); g.rect(G.padL, G.padT, w - G.padL - G.padR, h - G.padT - G.padB); g.clip();
  g.setLineDash([5, 4]); g.textAlign = "right"; g.textBaseline = "bottom";
  for (const r of grp.refs) { const y = G.Y(r.y); g.strokeStyle = mute; g.beginPath(); g.moveTo(G.padL, y); g.lineTo(w - G.padR, y); g.stroke(); g.fillStyle = mute; g.fillText(r.label, w - G.padR - 4, y - 2); }
  g.setLineDash([]);
  // lines (the one nearest to the pointer is drawn thicker, the others fade)
  let near = tv.hover != null ? nearestLine(grp, G, tv.hover) : null;
  ovFocus = near != null && grp.lines[near].joint ? grp.lines[near].joint : null;   // the overlay on the video marks this joint
  if (near == null && tv.hover == null && ovHover) { const k = grp.lines.findIndex((l) => l.joint === ovHover); near = k >= 0 ? k : null; }   // and the other way round
  grp.lines.forEach((l, k) => {
    if (tv.hidden.has(l.label)) return;
    g.strokeStyle = COLORS[k % COLORS.length]; g.lineJoin = "round";
    g.globalAlpha = near == null || near === k ? 1 : 0.35; g.lineWidth = near === k ? 2.6 : 1.8;
    g.beginPath();
    let pen = false;
    for (let i = Math.max(0, G.i0 - 1); i <= Math.min(l.data.length - 1, G.i1 + 1); i++) {
      if (!Number.isFinite(l.data[i])) { pen = false; continue; }   // a gap where there is no value
      pen ? g.lineTo(G.X(i), G.Y(l.data[i])) : g.moveTo(G.X(i), G.Y(l.data[i])); pen = true;
    }
    g.stroke();
  });
  g.globalAlpha = 1;
  g.restore();
  // video cursor
  const v = $("video");
  if (v && v.style.display !== "none" && v.duration) {
    const i = Math.min(n - 1, (v.currentTime % (n * dt)) / dt);
    if (i >= G.i0 && i <= G.i1) { const x = G.X(i); g.strokeStyle = ink; g.globalAlpha = 0.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(x, G.padT); g.lineTo(x, h - G.padB); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
  }
  // drag selection
  if (tv.drag && tv.drag.moved) { g.fillStyle = "rgba(59,130,246,.18)"; g.fillRect(Math.min(tv.drag.x0, tv.drag.x1), G.padT, Math.abs(tv.drag.x1 - tv.drag.x0), h - G.padT - G.padB); }
  // hover crosshair + tooltip
  if (tv.hover != null && !(tv.drag && tv.drag.moved)) {
    const i = G.iAt(tv.hover), x = G.X(i);
    g.strokeStyle = mute; g.beginPath(); g.moveTo(x, G.padT); g.lineTo(x, h - G.padB); g.stroke();
    const rows = grp.lines.map((l, k) => ({l, k})).filter(({l}) => !tv.hidden.has(l.label) && l.data[i] != null);
    rows.forEach(({l, k}) => { if (!Number.isFinite(l.data[i])) return; g.fillStyle = COLORS[k % COLORS.length]; g.beginPath(); g.arc(x, G.Y(l.data[i]), 3.5, 0, 7); g.fill(); });
    const lines = [`${(i * dt).toFixed(1)} 秒`].concat(rows.map(({l}) => `${l.label}  ${Number.isFinite(l.data[i]) ? (+l.data[i]).toFixed(3) + (grp.unit ? " " + grp.unit : "") : "—"}`));
    const bw = Math.max(...lines.map((t) => g.measureText(t).width)) + 26, bh = lines.length * 16 + 8;
    let bx = x + 12; if (bx + bw > w - 4) bx = x - 12 - bw;
    const by = G.padT + 4;
    g.fillStyle = css.getPropertyValue("--card").trim(); g.strokeStyle = line; g.globalAlpha = 0.96;
    g.beginPath(); g.roundRect ? g.roundRect(bx, by, bw, bh, 6) : g.rect(bx, by, bw, bh); g.fill(); g.stroke(); g.globalAlpha = 1;
    g.textAlign = "left"; g.textBaseline = "middle";
    lines.forEach((t, j) => {
      if (j === 0) { g.fillStyle = mute; g.fillText(t, bx + 10, by + 12); return; }
      const k = rows[j - 1].k; g.fillStyle = COLORS[k % COLORS.length]; g.fillRect(bx + 10, by + j * 16 + 8, 8, 3);
      g.fillStyle = ink; g.fillText(t, bx + 22, by + j * 16 + 10);
    });
  }
  // zoom hint
  if (tv.range) { g.fillStyle = mute; g.textAlign = "right"; g.textBaseline = "top"; g.fillText("ダブルクリックで全体に戻す", w - G.padR, G.padT + 2); }
}

function nearestLine(grp, G, px) {
  const i = G.iAt(px); let best = null, bd = 1e9;
  grp.lines.forEach((l, k) => { if (tv.hidden.has(l.label) || l.data[i] == null) return; const d = Math.abs(G.Y(l.data[i]) - (tv.hoverY ?? 0)); if (d < bd) { bd = d; best = k; } });
  return best;
}

function drawLegend(grp) {
  const box = $("tracelegend");
  box.textContent = "";
  grp.lines.forEach((l, k) => {
    const sw = el("span", null, "lg" + (tv.hidden.has(l.label) ? " off" : ""));
    const i = el("i"); i.style.background = COLORS[k % COLORS.length]; sw.appendChild(i); sw.appendChild(document.createTextNode(l.label));
    sw.onclick = () => { tv.hidden.has(l.label) ? tv.hidden.delete(l.label) : tv.hidden.add(l.label); drawLegend(grp); paintTrace(); };
    box.appendChild(sw);
  });
}

function bindTrace() {
  const cv = $("trace");
  const px = (e) => e.clientX - cv.getBoundingClientRect().left;
  const geom = () => { const grp = trace && trace.groups && trace.groups[Number($("tracegroup").value) || 0]; return grp ? {grp, G: chartGeom(cv, grp, Math.max(...grp.lines.map((l) => l.data.length)))} : null; };
  cv.onmousemove = (e) => {
    tv.hover = px(e); tv.hoverY = e.clientY - cv.getBoundingClientRect().top;
    if (tv.drag) { tv.drag.x1 = tv.hover; if (Math.abs(tv.drag.x1 - tv.drag.x0) > 5) tv.drag.moved = true; }
    paintTrace();
  };
  cv.onmouseleave = () => { tv.hover = null; tv.drag = null; paintTrace(); };
  cv.onmousedown = (e) => { tv.drag = {x0: px(e), x1: px(e), moved: false}; };
  cv.onmouseup = (e) => {
    const d = tv.drag, c = geom(); tv.drag = null;
    if (!c || !d) return;
    if (d.moved) {
      const a = c.G.iAt(Math.min(d.x0, d.x1)), b = c.G.iAt(Math.max(d.x0, d.x1));
      if (b - a >= 3) tv.range = [a, b];
    } else {   // a click seeks the video to that moment
      const v = $("video"), t = c.G.iAt(px(e)) * trace.dt;
      if (v && v.style.display !== "none" && v.duration) v.currentTime = Math.min(t, v.duration - 0.01);
    }
    paintTrace();
  };
  cv.ondblclick = () => { tv.range = null; paintTrace(); };
}

async function drawTrace(lp) {
  const msg = $("tracemsg"), sel = $("tracegroup"), cv = $("trace");
  if (!lp) { cv.getContext("2d").clearRect(0, 0, cv.width, cv.height); $("tracelegend").textContent = ""; $("tracegroup").textContent = ""; trace = null; traceKey = ""; msg.textContent = "録画が済むと、ここに動きのグラフが出ます"; return; }
  lastTraceLoop = lp;
  const key = `${exp}/${lp.loop}`;
  if (traceFailAt[key] && Date.now() - traceFailAt[key] < 2500) return;   // loading failed a moment ago: try again shortly
  if (traceKey !== key) {
    traceKey = key; trace = null; msg.textContent = "読み込み中…"; tv.range = null; tv.hidden = new Set();
    cv.getContext("2d").clearRect(0, 0, cv.width, cv.height); $("tracelegend").textContent = ""; $("tracelegend").dataset.g = ""; sel.textContent = "";   // nothing of the previous loop stays
    try { trace = await api(`/api/trace?exp=${encodeURIComponent(exp)}&loop=${lp.loop}`); } catch (e) { trace = null; }
    if (traceKey !== key) return;
    if (!trace) { traceFailAt[key] = Date.now(); traceKey = ""; msg.textContent = "この周のグラフのデータを読み込めませんでした。再試行しています…"; return; }
    let groups = [];
    try { groups = traceGroups(trace); } catch (e) { console.error(e); }
    const keep = sel.value || new URLSearchParams(location.search).get("group");   // ?group=N: open on that item (screenshots)
    sel.textContent = "";
    groups.forEach((x, i) => { const o = el("option", x.unit ? `${x.title} [${x.unit}]` : x.title); o.value = i; sel.appendChild(o); });
    if (keep && groups.some((_, i) => String(i) === keep)) sel.value = keep;
    sel.onchange = () => { tv.range = null; tv.hidden = new Set(); drawTrace(lastTraceLoop); };
    if (trace) trace.groups = groups;
    bindTrace();
    const ha = new URLSearchParams(location.search).get("hoverat");   // ?hoverat=SECONDS: hover there (screenshots)
    if (ha !== null && groups.length) {
      const grp = groups[Number(sel.value) || 0], G = chartGeom(cv, grp, Math.max(...grp.lines.map((l) => l.data.length)));
      tv.hover = G.X(Number(ha) / trace.dt); tv.hoverY = G.Y(grp.lines[0].data[Math.round(Number(ha) / trace.dt)]);
    }
  }
  if (!trace || !trace.groups.length) { msg.textContent = "この周には記録された時系列がありません"; $("tracelegend").textContent = ""; return; }
  msg.textContent = trace.joint_names ? "" : "この周は関節角度が保存されていません（足・頭・膝の高さは表示できます）";
  const grp = trace.groups[Number(sel.value) || 0];
  if (!$("tracelegend").childNodes.length || $("tracelegend").dataset.g !== key + sel.value) { $("tracelegend").dataset.g = key + sel.value; drawLegend(grp); }
  paintTrace();
}

async function loadState() {
  const mine = exp;
  if (!mine) return;
  try {
    const s = await api("/api/state?exp=" + encodeURIComponent(mine));
    if (mine !== exp) return;   // the person chose another experiment (or none) while this was loading: drop the answer
    offset = s.server_time - Date.now() / 1000;
    S = s;
    render();
  } catch (e) { /* keep last state */ }
}

async function loadExps(keep) {
  const list = await api("/api/experiments");
  const sel = $("expsel"); sel.textContent = "";
  const blank = el("option", "（選んでください）"); blank.value = ""; sel.appendChild(blank);
  for (const x of list) { const o = el("option", `${x.exp}  ${x.instruction}`); o.value = x.exp; sel.appendChild(o); }
  if (keep && list.some((x) => x.exp === exp)) sel.value = exp;   // otherwise nothing is selected
  return list;
}

function startReplay() {
  if (!S) return;
  selLoop = null; shownVideo = "";
  replay = {vt0: S.t0, wall0: Date.now(), speed: Number($("speed").value)};
  render();
}
function stopReplay() { replay = null; selLoop = null; shownVideo = ""; render(); }

const EMPTY_MSG = "指示を入力して学習を始めるか、「過去の実験を開く」から選んでください";
function pickExp(name, msg) {
  if (!name) { exp = null; replay = null; selLoop = null; S = null; trace = null; traceKey = ""; $("expsel").value = ""; clearView(EMPTY_MSG); $("modebadge").textContent = ""; $("replaybtn").hidden = true; $("replaybar").hidden = true; $("stopbtn").hidden = true; $("nowline").hidden = true; document.title = "Genesis Dojo"; return; }
  $("replaybtn").hidden = false;
  exp = name; replay = null; selLoop = null; S = null; trace = null; traceKey = "";
  clearView(msg);
  loadState();
}

$("expsel").onchange = (e) => pickExp(e.target.value);
$("speed").onchange = () => { if (replay) { const vt = virtualNow(); replay = {vt0: vt, wall0: Date.now(), speed: Number($("speed").value)}; } };
$("replaybtn").onclick = startReplay;
$("replaystop").onclick = stopReplay;
$("stopbtn").onclick = async () => {
  if (!exp || !confirm("学習を強制終了しますか？\nいま実行中の周は完了しません（これまでに終わった周の結果は残ります）。")) return;
  const b = $("stopbtn"); b.disabled = true; b.textContent = "停止しています…";
  try { await api("/api/stop", {method: "POST", headers: {"Content-Type": "application/json", "X-Requested-With": "dashboard"}, body: JSON.stringify({exp})}); }
  catch (e) { alert((e && e.error) || "止められませんでした"); }
  b.disabled = false; b.textContent = "■ 強制終了"; loadState();
};
["fixblock", "chgblock"].forEach((id) => { $(id).onmouseenter = () => hlStage(true); $(id).onmouseleave = () => hlStage(false); });
$("dclose").onclick = closeDrawer;
$("mclose").onclick = closeDrawer;
$("backdrop").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
new ResizeObserver(() => { if (trace && trace.groups) paintTrace(); }).observe($("tracewrap"));

const toggleOverlay = () => { ovOn = !ovOn; updateOvBtn(); };
const toggleScale = () => { ovScale = !ovScale; updateOvBtn(); };
const toggleView = () => { viewFree = !viewFree; updateOvBtn(); };
$("ovbtn").onclick = toggleOverlay; $("scalebtn").onclick = toggleScale; $("viewbtn").onclick = toggleView;
document.addEventListener("keydown", (e) => {
  if (!ov || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === "j") toggleOverlay(); else if (k === "g") toggleScale(); else if (k === "v") toggleView();
});
// free camera: drag = orbit, wheel = zoom, double-click = reset
{
  const cv = $("overlay");
  cv.addEventListener("mousedown", (e) => { if (viewFree) { fdrag = {x: e.clientX, y: e.clientY}; cv.style.cursor = "grabbing"; } });
  window.addEventListener("mousemove", (e) => {
    if (!fdrag) return;
    fc.az -= (e.clientX - fdrag.x) * 0.008; fc.el = Math.max(-0.15, Math.min(1.5, fc.el + (e.clientY - fdrag.y) * 0.008));
    fdrag.x = e.clientX; fdrag.y = e.clientY;
  });
  window.addEventListener("mouseup", () => { fdrag = null; if (viewFree) cv.style.cursor = "grab"; });
  cv.addEventListener("wheel", (e) => { if (!viewFree) return; e.preventDefault(); fc.dist = Math.max(0.8, Math.min(12, fc.dist * Math.exp(e.deltaY * 0.001))); }, {passive: false});
  cv.addEventListener("dblclick", () => { if (viewFree) fc = {...FREE0}; });
  $("fplay").onclick = () => { const v = $("video"); v.paused ? v.play() : v.pause(); };
  $("fslider").oninput = (e) => { fdrag = {slider: true}; const v = $("video"); v.pause(); v.currentTime = Number(e.target.value) * (ov ? ov.dt : 0.02); };
  $("fslider").onchange = () => { fdrag = null; };
}
$("videobox").onmousemove = (e) => { const r = $("videobox").getBoundingClientRect(); ovMouse = [e.clientX - r.left, e.clientY - r.top]; };
$("videobox").onmouseleave = () => { ovMouse = null; };

// ---- creating the success spec on the dashboard: draft (AI) -> the person edits and approves -> run ----
let metricList = [], baseList = [];
const KIND = (name) => (metricList.find((m) => m.name === name) || {}).kind;
const newExpName = () => "run" + new Date().toISOString().replace(/\D/g, "").slice(2, 14);

function openDrawer(title, step) {   // the authoring dialog (centred, with its own steps)
  $("mtitle").textContent = title; $("mbody").textContent = "";
  [...$("msteps").children].forEach((li, i) => li.className = i < step ? "done" : i === step ? "on" : "");
  $("modal").hidden = false; $("backdrop").hidden = false;
  return $("mbody");
}
const setStep = (n) => [...$("msteps").children].forEach((li, i) => { li.className = i < n ? "done" : i === n ? "on" : ""; });

async function startRun(instruction, task) {
  const name = newExpName();
  const item = await api("/api/run", {method: "POST", headers: {"Content-Type": "application/json", "X-Requested-With": "dashboard"}, body: JSON.stringify({exp: name, instruction, task})});
  return {name, item};
}
async function afterStart(name) {
  setStep(3);
  closeDrawer();
  $("formmsg").textContent = "学習を始めました";
  await new Promise((r) => setTimeout(r, 1500));
  await loadExps(false); selectExp(name, "新しい実験"); pickExp(name, "学習の準備をしています…");
}
// Select an experiment in the menu, adding it when the list does not have it yet (a run that has just started).
function selectExp(name, label) {
  const sel = $("expsel");
  if (![...sel.options].some((o) => o.value === name)) { const o = el("option", `${name}  ${label}`); o.value = name; sel.appendChild(o); }
  exp = name; sel.value = name;
}
const post = (path, body) => api(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Requested-With": "dashboard"}, body: JSON.stringify(body)});

async function authorFlow(instruction, forceNew) {
  const body = openDrawer("成功条件を決める", 0);
  body.appendChild(el("p", `指示: 「${instruction}」`));
  const wait = el("p", forceNew ? "AIが、新しい成功条件の草案を作っています…（30秒〜1分）" : "既存の承認済みの条件に当てはまるかを調べています…", "muted");
  body.appendChild(wait);
  let r;
  try { r = await post("/api/draft", {instruction, force_new: !!forceNew}); }
  catch (e) { wait.textContent = (e && e.error) || "草案を作れませんでした。もう一度お試しください"; return; }
  wait.remove();
  setStep(1);
  if (r.reuse) return renderReuse(body, instruction, r.reuse);
  if (r.cannot_measure) { body.appendChild(el("p", "この指示は、いまの測り方では成功条件にできません。", "bad")); body.appendChild(el("p", r.cannot_measure)); return; }
  await renderDraft(body, instruction, r.draft, r.errors || [], r.hint);
}

function renderReuse(body, instruction, c) {
  body.appendChild(el("p", `この指示は、承認済みの成功条件「${c.name}」に当てはまります。`));
  body.appendChild(el("p", `${c.description}（基準 ${c.criteria} 項目。人が承認して固定済み）`, "muted"));
  const act = el("div", null, "actions");
  const use = el("button", "この条件で学習を始める");
  use.onclick = async () => { use.disabled = true; try { const {name} = await startRun(instruction, c.task); await afterStart(name); } catch (e) { body.appendChild(el("p", (e && e.error) || "開始できません", "bad")); use.disabled = false; } };
  const mk = el("button", "新しい成功条件を作る", "ghost");
  mk.onclick = () => authorFlow(instruction, true);
  act.appendChild(use); act.appendChild(mk); body.appendChild(act);
}

function hintCard(body, hint) {
  if (!hint) return;
  const det = el("details"); det.appendChild(el("summary", `設計ヒント「${hint.name}」が、報酬を書くAIに渡されます（人が事前に書いた助言。AI採点には見せません）`));
  det.appendChild(el("pre", hint.text)); body.appendChild(det);
}

async function renderDraft(body, instruction, d, errors, hint) {
  if (!metricList.length) { try { metricList = await api("/api/metrics"); } catch (e) { metricList = []; } }
  if (!baseList.length) { try { baseList = await api("/api/bases"); } catch (e) { baseList = []; } }
  body.appendChild(el("p", "AIが作った草案です。内容を確認して、必要なら直してから承認してください。承認すると、この条件が「成功条件」として固定され、学習中は変更されません（承認は、あなたの操作でのみ行われます）。"));
  const field = (label, input) => { const f = el("label", label, "field"); f.appendChild(input); body.appendChild(f); return input; };
  const name = el("input"); name.type = "text"; name.value = d.name || ""; field("芸の名前", name);
  const tid = el("input"); tid.type = "text"; tid.value = d.task_id || ""; field("識別名（半角小文字と数字と _ ）", tid);
  const base = el("select"); baseList.forEach((b) => { const o = el("option", `${b.path}（録画 ${b.record_s} 秒）${b.note ? " — " + b.note : ""}`); o.value = b.path; base.appendChild(o); });
  base.value = d.base; field("学習の土台（姿勢や録画の設定）", base);
  const w0 = el("input"); w0.type = "number"; w0.step = "0.5"; w0.value = (d.window_s || [3, 6])[0];
  const w1 = el("input"); w1.type = "number"; w1.step = "0.5"; w1.value = (d.window_s || [3, 6])[1];
  const wf = el("div", null, "rule"); wf.appendChild(w0); wf.appendChild(el("span", "秒 〜")); wf.appendChild(w1); wf.appendChild(el("span", "秒"));
  field("判定する時間帯（録画のうち、この間の動きで判定する）", wf);
  if (d.rationale) body.appendChild(el("p", "AIの説明: " + d.rationale, "note"));
  hintCard(body, hint);
  body.appendChild(el("p", "成功の基準（動画から測る数値の条件。すべて満たすと数値判定が合格）", "note"));
  const list = el("div"); body.appendChild(list);
  const rows = [];
  const addRow = (c) => rows.push(critRow(list, c, rows));
  (d.criteria || []).forEach(addRow);
  const add = el("select"); add.appendChild(el("option", "＋ 基準を追加…"));
  metricList.forEach((m) => { const o = el("option", `${m.name} — ${m.text}`.slice(0, 110)); o.value = m.name; add.appendChild(o); });
  add.onchange = () => { if (add.value && add.selectedIndex > 0) { addRow({metric: add.value, min: 0, min_fraction: 0.9, comment: ""}); } add.selectedIndex = 0; };
  body.appendChild(add);
  const err = el("div", errors.map((x) => "・" + x).join("\n"), "errors"); err.style.whiteSpace = "pre-wrap"; body.appendChild(err);
  const act = el("div", null, "actions");
  const ok = el("button", "確認した。承認して学習を始める");
  const no = el("button", "キャンセル", "ghost"); no.onclick = closeDrawer;
  ok.onclick = async () => {
    const draft = {task_id: tid.value.trim(), name: name.value.trim(), description: d.description, base: base.value,
                   window_s: [Number(w0.value), Number(w1.value)], criteria: rows.filter((r) => r.alive).map((r) => r.read())};
    ok.disabled = true; err.textContent = "";
    try { const nm = newExpName(); await post("/api/approve", {instruction, exp: nm, draft, hint_id: hint ? hint.id : null}); await afterStart(nm); }
    catch (e) { err.textContent = ((e && e.errors) || [(e && e.error) || "承認できません"]).map((x) => "・" + x).join("\n"); ok.disabled = false; }
  };
  act.appendChild(ok); act.appendChild(no); body.appendChild(act);
}

async function improveFlow() {
  const body = openDrawer("成功条件を見直す", 0);
  body.appendChild(el("p", `実験: 「${S.instruction}」`));
  const wait = el("p", "AIが、これまでの結果（各周の数値とAIの採点）から、基準の見直し案を作っています…（1〜2分）", "muted");
  body.appendChild(wait);
  let r;
  try { r = await post("/api/improve", {exp}); }
  catch (e) { wait.textContent = (e && e.error) || "見直し案を作れませんでした。もう一度お試しください"; return; }
  wait.remove(); setStep(1);
  if (!metricList.length) { try { metricList = await api("/api/metrics"); } catch (e) { metricList = []; } }
  const d = r.draft;
  body.appendChild(el("p", S.stop_reason === "success" ? "改善点の整理（AI）" : "診断（AI）", "note")); body.appendChild(el("p", r.diagnosis || "—"));
  if (r.rationale || d.rationale) { body.appendChild(el("p", "変更点と理由", "note")); body.appendChild(el("p", d.rationale || "")); }
  body.appendChild(el("p", "この基準で、これまでの録画（学習し直さずに）を判定すると", "note"));
  const prev = el("div"); body.appendChild(prev);
  const drawPreview = (rows) => {
    prev.textContent = "";
    const t = el("table"), hd = el("tr");
    ["周", "旧基準の数値判定", "AI採点", "新基準の数値判定", "AIと一致？"].forEach((x) => hd.appendChild(el("td", x, "muted")));
    t.appendChild(hd);
    for (const p of rows) {
      const tr = el("tr");
      tr.appendChild(el("td", `${p.loop}周目`)); tr.appendChild(el("td", p.old ? `${p.old} ${p.old_passed ? "合格" : "不合格"}` : "—"));
      tr.appendChild(el("td", p.ai_score == null ? "—" : `${p.ai_score}/10 ${p.ai_pass ? "合格" : "不合格"}`));
      tr.appendChild(el("td", `${p.n_passed}/${p.n_criteria} ${p.passed ? "合格" : "不合格"}`));
      tr.appendChild(el("td", p.agrees == null ? "—" : p.agrees ? "一致" : "不一致", p.agrees === false ? "bad" : ""));
      t.appendChild(tr);
    }
    prev.appendChild(rows.length ? t : el("p", "判定し直せる録画がありません", "muted"));
  };
  drawPreview(r.preview || []);
  const field = (label, input) => { const f = el("label", label, "field"); f.appendChild(input); body.appendChild(f); return input; };
  const name = el("input"); name.type = "text"; name.value = d.name || ""; field("新しい成功条件の名前", name);
  const tid = el("input"); tid.type = "text"; tid.value = d.task_id; field("識別名（新しい版。いまの成功条件は残ります）", tid);
  const w0 = el("input"); w0.type = "number"; w0.step = "0.5"; w0.value = (d.window_s || [3, 6])[0];
  const w1 = el("input"); w1.type = "number"; w1.step = "0.5"; w1.value = (d.window_s || [3, 6])[1];
  const wf = el("div", null, "rule"); wf.appendChild(w0); wf.appendChild(el("span", "秒 〜")); wf.appendChild(w1); wf.appendChild(el("span", "秒"));
  field("判定する時間帯", wf);
  body.appendChild(el("p", "成功の基準（直せます。「新規」「変更」は、いまの基準との違い）", "note"));
  const list = el("div"); body.appendChild(list);
  const old = r.old || [], rows = [];
  const same = (c) => old.some((o) => o.metric === c.metric && JSON.stringify(["between", "min", "max", "equals"].map((k) => o[k])) === JSON.stringify(["between", "min", "max", "equals"].map((k) => c[k])));
  const tag = (c) => same(c) ? "そのまま" : old.some((o) => o.metric === c.metric) ? "変更" : "新規";
  const addRow = (c) => rows.push(critRow(list, c, rows, tag(c)));
  (d.criteria || []).forEach(addRow);
  const add = el("select"); add.appendChild(el("option", "＋ 基準を追加…"));
  metricList.forEach((m) => { const o = el("option", `${m.name} — ${m.text}`.slice(0, 110)); o.value = m.name; add.appendChild(o); });
  add.onchange = () => { if (add.selectedIndex > 0) addRow({metric: add.value, min: 0, min_fraction: 0.9, comment: ""}); add.selectedIndex = 0; };
  body.appendChild(add);
  const current = () => ({task_id: tid.value.trim(), name: name.value.trim(), description: d.description, base: d.base, window_s: [Number(w0.value), Number(w1.value)], criteria: rows.filter((x) => x.alive).map((x) => x.read())});
  const recheck = el("button", "この基準で、これまでの録画を判定し直す", "ghost");
  const err = el("div", (r.errors || []).map((x) => "・" + x).join("\n"), "errors"); err.style.whiteSpace = "pre-wrap";
  recheck.onclick = async () => { err.textContent = ""; try { drawPreview((await post("/api/preview", {exp, draft: current()})).preview); } catch (e) { err.textContent = ((e && e.errors) || ["判定できません"]).map((x) => "・" + x).join("\n"); } };
  body.appendChild(el("div", null, "actions")); body.lastChild.appendChild(recheck); body.appendChild(err);
  const carry = el("input"); carry.type = "checkbox"; carry.checked = true;
  const carryLabel = el("label", null, "field"); carryLabel.style.flexDirection = "row"; carryLabel.style.alignItems = "center"; carryLabel.appendChild(carry);
  carryLabel.appendChild(el("span", `前の実験（${exp}）の、最良の周の報酬コードと、AIの指摘を、最初の周から報酬を書くAIに見せる（学習した重みは引き継ぎません）`)); body.appendChild(carryLabel);
  body.appendChild(el("p", "次の学習で、報酬を書く LLM に渡す助言（任意。直せます。AI採点には見せません。報告書には、助言を渡したことを書きます）", "note"));
  const hint = el("textarea"); hint.value = d.hint || ""; hint.rows = 3; hint.style.width = "100%"; body.appendChild(hint);
  const act = el("div", null, "actions");
  const go = async (start) => {
    err.textContent = ""; ok.disabled = only.disabled = true;
    try {
      const nm = newExpName();
      await post("/api/approve", {instruction: S.instruction, exp: nm, start, hint: hint.value, draft: current(), prev_exp: carry.checked ? exp : null});
      if (start) await afterStart(nm); else { closeDrawer(); $("formmsg").textContent = "新しい成功条件を承認しました（学習はしていません）"; }
    } catch (e) { err.textContent = ((e && e.errors) || [(e && e.error) || "承認できません"]).map((x) => "・" + x).join("\n"); ok.disabled = only.disabled = false; }
  };
  const ok = el("button", "確認した。承認して再学習する"); ok.onclick = () => go(true);
  const only = el("button", "承認だけする（学習はしない）", "ghost"); only.onclick = () => go(false);
  const no = el("button", "キャンセル", "ghost"); no.onclick = closeDrawer;
  act.appendChild(ok); act.appendChild(only); act.appendChild(no); body.appendChild(act);
}

const RULE_LABEL = {min: "以上", max: "以下", between: "範囲", equals: "一致"};
function critRow(list, c, rows, tag) {
  const box = el("div", null, "crit"), kind = KIND(c.metric);
  const x = el("button", "✕", "ghost x"); x.title = "この基準を外す"; box.appendChild(x);
  const cm = el("input"); cm.type = "text"; cm.value = c.comment || ""; cm.placeholder = "この基準の説明"; cm.className = "full"; box.appendChild(cm);
  const metric = (metricList.find((m) => m.name === c.metric) || {});
  box.appendChild(el("div", `${tag ? "【" + tag + "】 " : ""}${c.metric}${metric.text ? " — " + metric.text : ""}`, "full muted"));
  const rule = el("div", null, "rule"), sel = el("select");
  const cur = ["between", "min", "max", "equals"].find((k) => k in c) || "min";
  for (const k of Object.keys(RULE_LABEL)) { const o = el("option", RULE_LABEL[k]); o.value = k; sel.appendChild(o); }
  sel.value = cur; sel.style.width = "80px";
  const a = el("input"), b = el("input"); a.type = b.type = "number"; a.step = b.step = "any";
  a.value = cur === "between" ? c.between[0] : c[cur]; b.value = cur === "between" ? c.between[1] : "";
  const sync = () => { b.style.display = sel.value === "between" ? "" : "none"; }; sel.onchange = sync; sync();
  rule.appendChild(sel); rule.appendChild(a); rule.appendChild(b); box.appendChild(rule);
  const fr = el("input"); fr.type = "number"; fr.step = "0.05"; fr.min = "0"; fr.max = "1"; fr.value = c.min_fraction ?? 0.9;
  const frw = el("div", null, "rule"); frw.appendChild(el("span", "窓内で満たす割合")); frw.appendChild(fr);
  if (kind !== "series") frw.style.visibility = "hidden";
  box.appendChild(frw);
  const row = {alive: true, read() {
    const o = {metric: c.metric, comment: cm.value.trim()};
    o[sel.value] = sel.value === "between" ? [Number(a.value), Number(b.value)] : Number(a.value);
    if (kind === "series") o.min_fraction = Number(fr.value);
    return o;
  }};
  x.onclick = () => { row.alive = false; box.remove(); };
  list.appendChild(box);
  return row;
}

$("runform").onsubmit = async (e) => {
  e.preventDefault();
  const instruction = $("instruction").value.trim();
  if (instruction) authorFlow(instruction, false);
};

// Public demo: no form, but a card per recorded experiment that opens it and replays its course.
async function initDemo() {
  $("runform").hidden = true; $("demobox").hidden = false; $("speed").value = "16";
  const box = $("democards");
  for (const x of await api("/api/experiments")) {
    const b = el("button", x.label || x.instruction || x.exp, "ghost demo");
    b.onclick = async () => { selectExp(x.exp, x.instruction); pickExp(x.exp, "読み込み中…"); await loadState(); startReplay(); };
    box.appendChild(b);
  }
}

(async function init() {
  const cfg = await api("/api/config");
  $("runform").hidden = !cfg.can_run; canRun = !!cfg.can_run;
  try { catalog = await api("/api/catalog"); } catch (e) { catalog = []; }
  if (cfg.can_run && catalog.length) $("instruction").placeholder = "例: " + catalog.map((c) => (c.examples || [])[0]).filter(Boolean).slice(0, 3).join(" / ");
  const q = new URLSearchParams(location.search).get("exp");
  const qs = new URLSearchParams(location.search);   // ?scale=1&view=free: start with these on (screenshots)
  if (qs.get("scale")) ovScale = true;
  if (qs.get("view") === "free") viewFree = true;
  if (qs.get("az")) fc.az = Number(qs.get("az"));
  await loadExps(false);
  if (STATIC) await initDemo();
  if (q) { exp = q; $("expsel").value = q; }
  else { $("replaybtn").hidden = true; clearView(EMPTY_MSG); }   // nothing is opened until the person picks one or starts a run
  await loadState();
  if (new URLSearchParams(location.search).get("replay") && S) startReplay();
  const at = new URLSearchParams(location.search).get("at");  // freeze the replay at this many seconds (screenshots)
  if (at !== null && S) replay = {vt0: S.t0 + Number(at), wall0: Date.now(), speed: 0};
  const au = new URLSearchParams(location.search).get("author");   // ?author=TEXT: open the success-spec flow for that instruction (screenshots)
  if (au) authorFlow(au, true);
  if (qs.get("improve") && S) improveFlow();   // ?improve=1: open the revise flow (screenshots)
  const st = new URLSearchParams(location.search).get("stage");   // ?stage=reward: open that stage's drawer (screenshots)
  if (st && S) openStage(st, viewAt(S, replay ? virtualNow() : Date.now() / 1000).slice(-1)[0] || null);
  setInterval(() => { if (!replay) loadState(); }, 1500);
  setInterval(render, 500);
  setInterval(() => { if (trace && trace.groups) paintTrace(); }, 100);
})();
