// 3D "pop-out" pitch for the simulation. A pure CSS-3D scene: the pitch plane
// tilts on rotateX so it reads as depth, and players stand on it as billboarded
// cylinders. It is a *reader* of the same authoritative state the 2D pitch uses
// (state.players, state.ball, state.simulation) -- it owns no game logic. Kept
// dependency-free on purpose; no WebGL/Three, so it drops into the vendored
// stack with nothing new to load.

const HOME = "#38bdf8";
const AWAY = "#fb7185";
const BALL = "#fde047";
const THETA = 56;            // pitch tilt in degrees
const LERP = 0.28;           // position smoothing per frame

export class Sim3DView {
  constructor(mount) {
    this.mount = mount;
    this.pitchL = 105;
    this.pitchW = 68;
    this.state = null;
    this.visible = false;
    this.tokens = new Map();       // playerId -> { el, num, cur:{x,y} }
    this.ballCur = { x: 52.5, y: 34 };
    this._build();
    this._raf = requestAnimationFrame(() => this._frame());
  }

  _build() {
    this.root = document.createElement("div");
    this.root.className = "sim3d-root";
    this.root.style.cssText =
      "position:absolute;inset:0;display:none;z-index:40;overflow:hidden;" +
      "background:radial-gradient(130% 130% at 50% 0%,#0b1a12 0%,#06100b 60%,#020403 100%);" +
      `perspective:1200px;perspective-origin:50% 32%;--theta:${THETA}deg;`;

    // The tilting plane. Its own pixel box maps 1:1 to pitch metres via _layout.
    this.plane = document.createElement("div");
    this.plane.className = "sim3d-plane";
    this.plane.style.cssText =
      "position:absolute;left:50%;top:52%;transform-style:preserve-3d;" +
      "transform:translate(-50%,-50%) rotateX(var(--theta));" +
      "border-radius:6px;box-shadow:0 40px 120px #000a;";

    // Pitch surface + markings as an inline SVG child, so lines tilt with the
    // plane and stay crisp at any size.
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("viewBox", "0 0 1050 680");
    this.svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
    this.svg.innerHTML = this._pitchMarkup();
    this.plane.appendChild(this.svg);

    // Ground layer for the active pass line; players sit above it.
    this.laneLayer = document.createElement("div");
    this.laneLayer.style.cssText = "position:absolute;inset:0;transform-style:preserve-3d;";
    this.plane.appendChild(this.laneLayer);
    this.lane = document.createElement("div");
    this.lane.style.cssText =
      "position:absolute;height:5px;border-radius:3px;transform-origin:left center;" +
      "opacity:0;transition:opacity .2s;pointer-events:none;" +
      "box-shadow:0 0 12px currentColor;";
    this.laneLayer.appendChild(this.lane);

    this.playersLayer = document.createElement("div");
    this.playersLayer.style.cssText = "position:absolute;inset:0;transform-style:preserve-3d;";
    this.plane.appendChild(this.playersLayer);

    // Ball: a small billboarded disc with a ground shadow.
    this.ballShadow = document.createElement("div");
    this.ballShadow.style.cssText =
      "position:absolute;width:16px;height:9px;border-radius:50%;background:#0008;" +
      "filter:blur(2px);transform:translate(-50%,-50%);";
    this.ballEl = document.createElement("div");
    this.ballEl.style.cssText =
      `position:absolute;width:14px;height:14px;border-radius:50%;background:${BALL};` +
      "box-shadow:0 0 10px #fde04788,0 2px 4px #0009;" +
      "transform-origin:bottom center;transform:translate(-50%,-100%) rotateX(calc(-1*var(--theta)));";
    this.playersLayer.append(this.ballShadow, this.ballEl);

    this.plane.append();
    this.root.appendChild(this.plane);
    this.mount.appendChild(this.root);
  }

  _pitchMarkup() {
    const L = "stroke='#dcefe4' stroke-opacity='0.6' stroke-width='2.5' fill='none'";
    return `
      <rect x='0' y='0' width='1050' height='680' rx='6'
            fill='#0f4023'/>
      <defs>
        <linearGradient id='stripes' x1='0' x2='1' y1='0' y2='0'>
          <stop offset='0' stop-color='#0f4023'/><stop offset='1' stop-color='#12492a'/>
        </linearGradient>
      </defs>
      ${Array.from({ length: 10 }, (_, i) =>
        `<rect x='${i * 105}' y='0' width='105' height='680' fill='${i % 2 ? "#12492a" : "#0f4023"}'/>`).join("")}
      <rect x='6' y='6' width='1038' height='668' ${L}/>
      <line x1='525' y1='6' x2='525' y2='674' ${L}/>
      <circle cx='525' cy='340' r='91' ${L}/>
      <circle cx='525' cy='340' r='3' fill='#dcefe4' fill-opacity='0.6'/>
      <rect x='6' y='138' width='165' height='403' ${L}/>
      <rect x='6' y='248' width='55' height='183' ${L}/>
      <rect x='879' y='138' width='165' height='403' ${L}/>
      <rect x='989' y='248' width='55' height='183' ${L}/>
    `;
  }

  _layout() {
    const w = this.mount.clientWidth || 960;
    const h = this.mount.clientHeight || 600;
    // Fit the plane's footprint inside the viewport, leaving headroom for the
    // tilt to lift the far side and for cylinders to stand above the near edge.
    const pw = Math.min(w * 0.92, (h * 0.96) * (this.pitchL / this.pitchW) / 1.35);
    const ph = pw * (this.pitchW / this.pitchL);
    this.plane.style.width = `${pw}px`;
    this.plane.style.height = `${ph}px`;
    this._pw = pw;
    this._ph = ph;
  }

  px(x) { return (x / this.pitchL) * this._pw; }
  py(y) { return (y / this.pitchW) * this._ph; }

  show() {
    if (this.visible) return;
    this.visible = true;
    this.root.style.display = "block";
    this._layout();
    // Pop: start flat, settle into the tilt.
    this.plane.style.transition = "none";
    this.plane.style.transform = "translate(-50%,-50%) rotateX(0deg) scale(0.96)";
    requestAnimationFrame(() => {
      this.plane.style.transition = "transform .7s cubic-bezier(.16,.84,.34,1)";
      this.plane.style.transform = "translate(-50%,-50%) rotateX(var(--theta)) scale(1)";
    });
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.plane.style.transition = "transform .4s ease-in";
    this.plane.style.transform = "translate(-50%,-50%) rotateX(0deg) scale(0.96)";
    setTimeout(() => { if (!this.visible) this.root.style.display = "none"; }, 380);
  }

  applyState(state) {
    this.state = state;
    if (state?.pitch) { this.pitchL = state.pitch.length; this.pitchW = state.pitch.width; }
    const active = !!state?.simulation?.active;
    if (active) this.show(); else this.hide();
  }

  _token(p) {
    let t = this.tokens.get(p.id);
    if (t) return t;
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:absolute;transform-style:preserve-3d;pointer-events:none;";

    const shadow = document.createElement("div");
    shadow.style.cssText =
      "position:absolute;width:24px;height:12px;border-radius:50%;background:#0009;" +
      "filter:blur(3px);transform:translate(-50%,-50%);";

    const cyl = document.createElement("div");
    cyl.style.cssText =
      "position:absolute;width:26px;height:34px;border-radius:46% 46% 46% 46% / 22% 22% 22% 22%;" +
      "transform-origin:bottom center;" +
      "transform:translate(-50%,-100%) rotateX(calc(-1*var(--theta)));" +
      "display:flex;align-items:center;justify-content:center;" +
      "font:700 12px ui-monospace,monospace;color:#04121a;" +
      "box-shadow:inset 0 -8px 12px #0006, inset 0 6px 8px #fff6, 0 3px 6px #0007;";
    // Elliptical cap highlight, so the top reads as a cylinder lid.
    const cap = document.createElement("div");
    cap.style.cssText =
      "position:absolute;top:-4px;left:50%;width:26px;height:9px;border-radius:50%;" +
      "transform:translateX(-50%);background:#ffffff55;";
    cyl.appendChild(cap);
    const num = document.createElement("span");
    cyl.appendChild(num);

    wrap.append(shadow, cyl);
    this.playersLayer.appendChild(wrap);
    t = { wrap, cyl, num, cap, cur: { x: p.x, y: p.y }, key: "" };
    this.tokens.set(p.id, t);
    return t;
  }

  _frame() {
    this._raf = requestAnimationFrame(() => this._frame());
    if (!this.visible || !this.state) return;
    if (Math.abs((this.mount.clientWidth || 0) - (this._lastW || 0)) > 1) {
      this._lastW = this.mount.clientWidth;
      this._layout();
    }
    const sim = this.state.simulation || {};
    const attack = sim.attackingTeam;
    const carrier = sim.ballOwnerNumber;
    const seen = new Set();

    for (const p of this.state.players || []) {
      seen.add(p.id);
      const t = this._token(p);
      t.cur.x += (p.x - t.cur.x) * LERP;
      t.cur.y += (p.y - t.cur.y) * LERP;
      t.wrap.style.left = `${this.px(t.cur.x)}px`;
      t.wrap.style.top = `${this.py(t.cur.y)}px`;

      const isAttack = p.team === attack;
      const isCarrier = isAttack && p.number === carrier;
      const key = `${p.team}:${isAttack}:${isCarrier}`;
      if (t.key !== key) {
        t.key = key;
        const base = p.team === "home" ? HOME : AWAY;
        t.cyl.style.background =
          `linear-gradient(180deg,${base} 0%, ${this._shade(base, -34)} 100%)`;
        t.cyl.style.opacity = isAttack ? "1" : "0.82";
        t.cyl.style.outline = isCarrier ? "2px solid #fde047" : "none";
        t.cyl.style.boxShadow = isCarrier
          ? "inset 0 -8px 12px #0006, inset 0 6px 8px #fff6, 0 0 16px #fde047cc, 0 3px 6px #0007"
          : "inset 0 -8px 12px #0006, inset 0 6px 8px #fff6, 0 3px 6px #0007";
        t.num.textContent = p.number;
        t.wrap.style.zIndex = String(1000 + Math.round(t.cur.y));
      }
      t.wrap.style.zIndex = String(1000 + Math.round(this.py(t.cur.y)));
    }
    for (const [id, t] of this.tokens) {
      if (!seen.has(id)) { t.wrap.remove(); this.tokens.delete(id); }
    }

    const b = this.state.ball || { x: 52.5, y: 34 };
    this.ballCur.x += (b.x - this.ballCur.x) * (LERP + 0.1);
    this.ballCur.y += (b.y - this.ballCur.y) * (LERP + 0.1);
    const bx = this.px(this.ballCur.x), by = this.py(this.ballCur.y);
    this.ballEl.style.left = `${bx}px`; this.ballEl.style.top = `${by}px`;
    this.ballShadow.style.left = `${bx}px`; this.ballShadow.style.top = `${by}px`;
    this.ballEl.style.zIndex = "9999";

    this._drawLane(sim);
  }

  // The active step's intended pass, drawn on the ground between the two players.
  _drawLane(sim) {
    const steps = sim.steps || [];
    const step = steps.find((s) => s.status === "active");
    if (!step || step.type !== "pass" || step.toNumber == null) {
      this.lane.style.opacity = "0";
      return;
    }
    const attack = sim.attackingTeam;
    const from = (this.state.players || []).find((p) => p.team === attack && p.number === step.fromNumber);
    const to = (this.state.players || []).find((p) => p.team === attack && p.number === step.toNumber);
    if (!from || !to) { this.lane.style.opacity = "0"; return; }
    const x1 = this.px(from.x), y1 = this.py(from.y);
    const x2 = this.px(to.x), y2 = this.py(to.y);
    const len = Math.hypot(x2 - x1, y2 - y1);
    const ang = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
    const col = attack === "home" ? HOME : AWAY;
    this.lane.style.color = col;
    this.lane.style.background = `linear-gradient(90deg, ${col}00, ${col})`;
    this.lane.style.left = `${x1}px`;
    this.lane.style.top = `${y1}px`;
    this.lane.style.width = `${len}px`;
    this.lane.style.transform = `rotate(${ang}deg)`;
    this.lane.style.opacity = "0.85";
  }

  _shade(hex, pct) {
    const n = parseInt(hex.slice(1), 16);
    const clamp = (v) => Math.max(0, Math.min(255, v));
    const r = clamp((n >> 16) + pct), g = clamp(((n >> 8) & 255) + pct), b = clamp((n & 255) + pct);
    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
  }
}
