/* RepSight engine: pose -> features -> exercise match + rep counting.
   Pure JS, no DOM. Works in the browser (window.RepEngine) and Node (module.exports). */
(function (root) {
  'use strict';

  // MediaPipe pose landmark indices
  const LS = 11, RS = 12, LE = 13, RE = 14, LW = 15, RW = 16, LH = 23, RH = 24, LK = 25, RK = 26, LA = 27, RA = 28;

  const FEATURES = [
    { key: 'lElbow', label: 'Left elbow' },
    { key: 'rElbow', label: 'Right elbow' },
    { key: 'lShoulder', label: 'Left shoulder' },
    { key: 'rShoulder', label: 'Right shoulder' },
    { key: 'lHip', label: 'Left hip' },
    { key: 'rHip', label: 'Right hip' },
    { key: 'lKnee', label: 'Left knee' },
    { key: 'rKnee', label: 'Right knee' },
    { key: 'tilt', label: 'Torso tilt' },
    { key: 'wrist', label: 'Hand height' },
    { key: 'stance', label: 'Stance width' },
  ];
  const NF = FEATURES.length;
  const FI = Object.fromEntries(FEATURES.map((f, i) => [f.key, i]));

  const sub = (a, b) => [a.x - b.x, a.y - b.y, a.z - b.z];
  const len = (v) => Math.hypot(v[0], v[1], v[2]);
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  function angle(a, b, c) {
    const u = sub(a, b), v = sub(c, b);
    const d = len(u) * len(v) || 1e-9;
    return Math.acos(clamp((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / d, -1, 1)) * 180 / Math.PI;
  }

  /** world: 33 landmarks in metres, y pointing down (MediaPipe world landmarks). Returns 11 values in ~[0,1]. */
  function features(w) {
    const f = new Array(NF);
    f[0] = angle(w[LS], w[LE], w[LW]) / 180;
    f[1] = angle(w[RS], w[RE], w[RW]) / 180;
    f[2] = angle(w[LH], w[LS], w[LE]) / 180;
    f[3] = angle(w[RH], w[RS], w[RE]) / 180;
    f[4] = angle(w[LS], w[LH], w[LK]) / 180;
    f[5] = angle(w[RS], w[RH], w[RK]) / 180;
    f[6] = angle(w[LH], w[LK], w[LA]) / 180;
    f[7] = angle(w[RH], w[RK], w[RA]) / 180;
    const sh = mid(w[LS], w[RS]), hp = mid(w[LH], w[RH]), wr = mid(w[LW], w[RW]);
    const t = sub(sh, hp), tl = len(t) || 1e-9;
    f[8] = Math.acos(clamp(Math.abs(t[1]) / tl, 0, 1)) / (Math.PI / 2);
    f[9] = clamp(((sh.y - wr.y) / tl + 1.6) / 3.2, 0, 1);
    const hw = len(sub(w[LH], w[RH])) || 1e-9;
    f[10] = clamp(len(sub(w[LA], w[RA])) / hw / 5, 0, 1);
    return f;
  }

  /* ---------- Synthetic body (used for built-in templates and the demo) ---------- */
  const BODY = { torso: 0.5, thigh: 0.45, shin: 0.44, uarm: 0.3, farm: 0.27, hipW: 0.2, shW: 0.36, head: 0.24 };
  const rad = (d) => d * Math.PI / 180;
  // Directions: y down. up = (0,-1,0). Forward F = (0,0,-1) (towards camera). Lateral S = (1,0,0) for the left side.
  function dirSag(th) { return [0, -Math.cos(rad(th)), -Math.sin(rad(th))]; }
  function dirFront(th, s) { return [s * Math.sin(rad(th)), -Math.cos(rad(th)), 0]; }
  function seg(p, d, L) { return { x: p.x + d[0] * L, y: p.y + d[1] * L, z: p.z + d[2] * L }; }

  /** spec: {torso, legs:'s'|'f', arms:'s'|'f', L:{thigh,shin,uarm,farm}, R:{...}} angles in degrees from straight up */
  function bodyPose(spec, body) {
    const B = body || BODY;
    const pts = new Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));
    const hipMid = { x: 0, y: 0, z: 0 };
    const shMid = seg(hipMid, dirSag(spec.torso), B.torso);
    pts[0] = seg(shMid, dirSag(spec.torso), B.head);
    for (const [s, side] of [[1, 'L'], [-1, 'R']]) {
      const a = spec[side];
      const hip = { x: s * B.hipW / 2, y: 0, z: 0 };
      const sh = { x: shMid.x + s * B.shW / 2, y: shMid.y, z: shMid.z };
      const legDir = (th) => spec.legs === 'f' ? dirFront(th, s) : dirSag(th);
      const armDir = (th) => spec.arms === 'f' ? dirFront(th, s) : dirSag(th);
      const knee = seg(hip, legDir(a.thigh), B.thigh);
      const ankle = seg(knee, legDir(a.shin), B.shin);
      const elbow = seg(sh, armDir(a.uarm), B.uarm);
      const wrist = seg(elbow, armDir(a.farm), B.farm);
      const I = s === 1 ? { sh: LS, el: LE, wr: LW, hip: LH, kn: LK, an: LA } : { sh: RS, el: RE, wr: RW, hip: RH, kn: RK, an: RA };
      pts[I.sh] = sh; pts[I.el] = elbow; pts[I.wr] = wrist; pts[I.hip] = hip; pts[I.kn] = knee; pts[I.an] = ankle;
    }
    return pts;
  }
  function lerpSpec(a, b, t) {
    const o = { torso: a.torso + (b.torso - a.torso) * t, legs: a.legs, arms: a.arms, L: {}, R: {} };
    for (const s of ['L', 'R']) for (const k of ['thigh', 'shin', 'uarm', 'farm']) o[s][k] = a[s][k] + (b[s][k] - a[s][k]) * t;
    return o;
  }
  const both = (o) => ({ L: { ...o }, R: { ...o } });
  const pose = (torso, legs, arms, L, R) => ({ torso, legs, arms, L, R: R || { ...L } });

  // Built-in moves. color = bumper-plate colour name used by the UI.
  const STAND = { thigh: 180, shin: 180, uarm: 180, farm: 180 };
  const BUILTINS = [
    { id: 'squat', name: 'Squat', plate: 'red', tracks: 'Knees and hips',
      start: pose(0, 's', 's', { ...STAND }), mid: pose(40, 's', 's', { thigh: 100, shin: 200, uarm: 180, farm: 180 }),
      ignore: ['lElbow', 'rElbow', 'lShoulder', 'rShoulder', 'wrist'] },
    { id: 'pushup', name: 'Push-up', plate: 'blue', tracks: 'Elbows, body flat',
      start: pose(90, 's', 's', { thigh: 270, shin: 270, uarm: 180, farm: 180 }), mid: pose(95, 's', 's', { thigh: 268, shin: 268, uarm: 245, farm: 170 }),
      ignore: ['wrist', 'stance'] },
    { id: 'curl', name: 'Bicep curl', plate: 'yellow', tracks: 'Elbows, upper arms still',
      start: pose(0, 's', 's', { ...STAND }), mid: pose(0, 's', 's', { thigh: 180, shin: 180, uarm: 170, farm: 15 }),
      ignore: ['stance'] },
    { id: 'press', name: 'Shoulder press', plate: 'green', tracks: 'Elbows and shoulders, overhead',
      start: pose(0, 's', 'f', { thigh: 180, shin: 180, uarm: 90, farm: 5 }), mid: pose(0, 's', 'f', { thigh: 180, shin: 180, uarm: 12, farm: 8 }),
      ignore: ['stance', 'lKnee', 'rKnee'] },
    { id: 'lateral', name: 'Lateral raise', plate: 'white', tracks: 'Shoulders, arms long',
      start: pose(0, 's', 'f', { thigh: 180, shin: 180, uarm: 172, farm: 170 }), mid: pose(0, 's', 'f', { thigh: 180, shin: 180, uarm: 92, farm: 86 }),
      ignore: ['stance'] },
    { id: 'jack', name: 'Jumping jack', plate: 'red', tracks: 'Arms overhead, feet apart',
      start: pose(0, 'f', 'f', { thigh: 180, shin: 180, uarm: 172, farm: 172 }), mid: pose(0, 'f', 'f', { thigh: 158, shin: 160, uarm: 15, farm: 12 }),
      ignore: [] },
    { id: 'lunge-l', group: 'lunge', name: 'Lunge', plate: 'blue', tracks: 'Front knee, back knee',
      start: pose(0, 's', 's', { ...STAND }), mid: pose(5, 's', 's', { thigh: 95, shin: 185, uarm: 180, farm: 180 }, { thigh: 215, shin: 285, uarm: 180, farm: 180 }),
      ignore: ['lElbow', 'rElbow', 'lShoulder', 'rShoulder', 'wrist', 'stance'] },
    { id: 'lunge-r', group: 'lunge', name: 'Lunge', plate: 'blue', tracks: 'Front knee, back knee', hidden: true,
      start: pose(0, 's', 's', { ...STAND }), mid: pose(5, 's', 's', { thigh: 215, shin: 285, uarm: 180, farm: 180 }, { thigh: 95, shin: 185, uarm: 180, farm: 180 }),
      ignore: ['lElbow', 'rElbow', 'lShoulder', 'rShoulder', 'wrist', 'stance'] },
    { id: 'deadlift', name: 'Deadlift', plate: 'yellow', tracks: 'Hip hinge, knees soft',
      start: pose(0, 's', 's', { ...STAND }), mid: pose(75, 's', 's', { thigh: 168, shin: 192, uarm: 180, farm: 180 }),
      ignore: ['lElbow', 'rElbow', 'stance'] },
    { id: 'situp', name: 'Sit-up', plate: 'green', tracks: 'Torso lifting from the floor',
      start: pose(270, 's', 's', { thigh: 45, shin: 140, uarm: 270, farm: 90 }), mid: pose(330, 's', 's', { thigh: 45, shin: 140, uarm: 330, farm: 150 }),
      ignore: ['lElbow', 'rElbow', 'lShoulder', 'rShoulder', 'wrist', 'stance'] },
  ];

  function buildWeights(start, mid, ignore) {
    const d = start.map((s, i) => Math.abs(mid[i] - s));
    const maxD = Math.max(...d) || 1e-9;
    return d.map((di, i) => (ignore && ignore.includes(FEATURES[i].key)) ? 0 : 0.3 + 1.7 * di / maxD);
  }

  function makeMove(def) {
    const start = def.startF || features(bodyPose(def.start));
    const midF = def.midF || features(bodyPose(def.mid));
    const w = def.weights || buildWeights(start, midF, def.ignore);
    const d = midF.map((m, i) => m - start[i]);
    let dd = 0, wsum = 0;
    for (let i = 0; i < NF; i++) { dd += w[i] * d[i] * d[i]; wsum += w[i]; }
    return { id: def.id, group: def.group || def.id, name: def.name, plate: def.plate, tracks: def.tracks, custom: !!def.custom,
      hidden: !!def.hidden, startF: start, midF, weights: w, d, dd: dd || 1e-9, wsum: wsum || 1e-9, spec: def.start ? { start: def.start, mid: def.mid } : null };
  }

  /** progress along the move (0 = start pose, 1 = peak pose) and weighted RMS distance from the move's path. */
  function evalMove(m, f) {
    let num = 0;
    for (let i = 0; i < NF; i++) num += m.weights[i] * (f[i] - m.startF[i]) * m.d[i];
    const p = num / m.dd;
    const pc = clamp(p, 0, 1);
    let e = 0;
    for (let i = 0; i < NF; i++) { const r = f[i] - (m.startF[i] + pc * m.d[i]); e += m.weights[i] * r * r; }
    return { p, dist: Math.sqrt(e / m.wsum) };
  }

  /* ---------- Teaching a custom move from a recorded clip ---------- */
  function teach(frames, name, id) {
    // frames: array of feature vectors
    if (frames.length < 20) return { error: 'The clip is too short. Record at least 3 full reps.' };
    const n = frames.length, mean = new Array(NF).fill(0);
    for (const f of frames) for (let i = 0; i < NF; i++) mean[i] += f[i] / n;
    const C = Array.from({ length: NF }, () => new Array(NF).fill(0));
    for (const f of frames) for (let i = 0; i < NF; i++) for (let j = 0; j < NF; j++) C[i][j] += (f[i] - mean[i]) * (f[j] - mean[j]) / n;
    let v = new Array(NF).fill(1 / Math.sqrt(NF));
    for (let it = 0; it < 60; it++) {
      const nv = C.map((row) => row.reduce((s, c, j) => s + c * v[j], 0));
      const nl = Math.hypot(...nv) || 1e-9; v = nv.map((x) => x / nl);
    }
    const proj = frames.map((f) => f.reduce((s, x, i) => s + (x - mean[i]) * v[i], 0));
    const sorted = [...proj].sort((a, b) => a - b);
    const q = (t) => sorted[Math.floor(t * (n - 1))];
    const lo = q(0.15), hi = q(0.85);
    const avgOf = (pred) => { const a = new Array(NF).fill(0); let c = 0; frames.forEach((f, k) => { if (pred(proj[k])) { c++; for (let i = 0; i < NF; i++) a[i] += f[i]; } }); return a.map((x) => x / (c || 1)); };
    const lowF = avgOf((p) => p <= lo), highF = avgOf((p) => p >= hi);
    // The end the clip spends its first moments near is the resting (start) position.
    const head = proj.slice(0, Math.max(5, Math.floor(n * 0.1)));
    const headMed = [...head].sort((a, b) => a - b)[Math.floor(head.length / 2)];
    const startIsLow = Math.abs(headMed - lo) <= Math.abs(headMed - hi);
    const startF = startIsLow ? lowF : highF, midF = startIsLow ? highF : lowF;
    const amp = Math.sqrt(startF.reduce((s, x, i) => s + (midF[i] - x) ** 2, 0));
    if (amp < 0.12) return { error: 'Not enough movement found. Make sure your whole body is in frame and do full reps.' };
    const m = makeMove({ id: id || ('custom-' + Date.now().toString(36)), name, plate: 'custom', tracks: topJoints(startF, midF), custom: true, startF, midF, weights: buildWeights(startF, midF, []) });
    // count reps in the teaching clip for feedback
    const eng = new Engine([m]);
    frames.forEach((f, k) => eng.update(f, k / 30));
    eng.finish(n / 30);
    const reps = eng.sets.reduce((s, x) => s + x.reps, 0);
    return { move: m, repsSeen: reps };
  }
  function topJoints(a, b) {
    const d = a.map((x, i) => ({ i, d: Math.abs(b[i] - x) })).sort((p, q) => q.d - p.d).slice(0, 2);
    return d.map((x) => FEATURES[x.i].label).join(', ');
  }

  /* ---------- Engine: windowed matching, set detection, rep counting ---------- */
  const WIN = 3.2;       // seconds of motion considered when matching
  const DMAX = 0.19;     // distance at which a match is considered worthless
  const ON = 0.42;       // score needed to start a set
  const HOLD = 0.5;      // seconds a new winner must lead before switching
  const IDLE_END = 2.6;  // seconds without a match before a set closes

  class Engine {
    constructor(moves) { this.setMoves(moves); this.reset(); }
    setMoves(moves) {
      this.moves = moves;
      this.groups = [...new Set(moves.map((m) => m.group))];
      if (this.state) this.reset();
    }
    reset() {
      this.buf = [];
      this.ema = null;
      this.state = Object.fromEntries(this.moves.map((m) => [m.id, { phase: 'top', ps: null, repDist: [], reps: [] }]));
      this.active = null; this.cand = null; this.candSince = 0; this.lastGood = 0;
      this.current = null; this.sets = []; this.events = [];
      this.scores = {}; this.t = 0;
    }
    update(rawF, t) {
      this.t = t;
      const a = 0.55;
      this.ema = this.ema ? this.ema.map((v, i) => v + a * (rawF[i] - v)) : rawF.slice();
      const f = this.ema;
      const ev = {};
      for (const m of this.moves) {
        const r = evalMove(m, f); ev[m.id] = r;
        const st = this.state[m.id];
        st.ps = st.ps == null ? r.p : st.ps + 0.6 * (r.p - st.ps);
        st.repDist.push(r.dist);
        if (st.phase === 'top' && st.ps > 0.68) st.phase = 'bottom';
        else if (st.phase === 'bottom' && st.ps < 0.32) {
          st.phase = 'top';
          const md = st.repDist.reduce((s, x) => s + x, 0) / st.repDist.length;
          if (md < DMAX * 0.75) { st.reps.push(t); this.onRep(m, t); }
          st.repDist = [];
        }
        if (st.phase === 'top' && st.ps < 0.32) st.repDist = [r.dist]; // a rep is measured from the moment it leaves the start pose
      }
      this.buf.push({ t, ev });
      while (this.buf.length && this.buf[0].t < t - WIN) this.buf.shift();
      // window scores per group
      const gs = {};
      for (const m of this.moves) {
        let dsum = 0, pmin = Infinity, pmax = -Infinity;
        for (const b of this.buf) { const r = b.ev[m.id]; dsum += r.dist; const p = clamp(r.p, -0.3, 1.3); if (p < pmin) pmin = p; if (p > pmax) pmax = p; }
        const md = dsum / this.buf.length, cov = pmax - pmin;
        const s = clamp(1 - md / DMAX, 0, 1) * clamp((cov - 0.3) / 0.4, 0, 1);
        if (!(m.group in gs) || s > gs[m.group].score) gs[m.group] = { score: s, dist: md, cov, id: m.id };
      }
      this.scores = gs;
      let best = null;
      for (const g in gs) if (!best || gs[g].score > gs[best].score) best = g;
      const bestScore = best ? gs[best].score : 0;
      if (bestScore >= ON) {
        if (best !== this.cand) { this.cand = best; this.candSince = t; }
        if (best === this.active) this.lastGood = t;
        else if (!this.active || (t - this.candSince >= HOLD && bestScore > (gs[this.active] ? gs[this.active].score : 0) + 0.08)) this.activate(best, t);
      } else this.cand = null;
      if (this.active && gs[this.active] && gs[this.active].score >= ON * 0.7) this.lastGood = t;
      if (this.active && t - this.lastGood > IDLE_END) this.closeSet(t);
      return this.snapshot();
    }
    groupName(g) { const m = this.moves.find((x) => x.group === g); return m ? m.name : g; }
    activate(g, t) {
      if (this.current) this.closeSet(t);
      this.active = g; this.lastGood = t;
      const ids = this.moves.filter((m) => m.group === g).map((m) => m.id);
      const back = [];
      for (const id of ids) for (const rt of this.state[id].reps) if (rt >= t - WIN - 1) back.push(rt);
      this.current = { group: g, name: this.groupName(g), start: Math.min(t, ...back), end: t, reps: back.length, repTimes: back.sort((a, b) => a - b) };
      this.events.push({ type: 'start', group: g, t });
    }
    onRep(m, t) {
      if (this.current && this.current.group === m.group) {
        const last = this.current.repTimes[this.current.repTimes.length - 1];
        if (last != null && t - last < 0.35) return; // same rep seen by mirrored variant
        this.current.reps++; this.current.repTimes.push(t); this.current.end = t;
        this.events.push({ type: 'rep', group: m.group, t, reps: this.current.reps });
      }
    }
    closeSet(t) {
      if (this.current && this.current.reps > 0) this.sets.push({ ...this.current, end: Math.max(this.current.end, this.current.repTimes.at(-1) || t) });
      this.events.push({ type: 'end', group: this.active, t });
      this.current = null; this.active = null;
    }
    finish(t) { if (this.active) this.closeSet(t); }
    snapshot() {
      const s = this.active ? this.scores[this.active] : null;
      return { active: this.active, name: this.active ? this.groupName(this.active) : null, reps: this.current ? this.current.reps : 0,
        match: s ? s.score : 0, cand: this.cand, candScore: this.cand ? this.scores[this.cand].score : 0, scores: this.scores };
    }
  }

  const api = { FEATURES, FI, features, bodyPose, lerpSpec, BUILTINS, BODY, makeMove, evalMove, teach, Engine,
    builtinMoves: () => BUILTINS.map(makeMove), consts: { WIN, DMAX, ON } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.RepEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
