const PITCH_L = 105;
const PITCH_W = 68;

const COL_GRASS = 0x0d3b1a;   
const COL_LINE = 0xe8f5e9;
const COL_HOME = 0x38bdf8;
const COL_AWAY = 0xfb7185;
const COL_BALL = 0xfde047;
const COL_CALIB = 0xff00ff;   

export class PitchRenderer {
  constructor(el, { showCalibration = true } = {}) {
    this.el = el;
    this.showCalibration = showCalibration;
    this.state = null;
    this.fps = 0;

    this.onDragStart = () => {};
    this.onDragMove = () => {};
    this.onDragEnd = () => {};

    this.app = new PIXI.Application({
      resizeTo: el,
      backgroundColor: 0x000000,
      antialias: true,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
    });
    el.appendChild(this.app.view);

    this.pitchLayer = new PIXI.Graphics();
    this.overlayLayer = new PIXI.Graphics();
    this.playersLayer = new PIXI.Container();
    this.cursorsLayer = new PIXI.Container();
    this.app.stage.addChild(this.pitchLayer, this.overlayLayer,
                            this.playersLayer, this.cursorsLayer);

    this.sprites = new Map();  
    this.cursorSprites = new Map();
    this.ball = new PIXI.Graphics();
    this.playersLayer.addChild(this.ball);
    this.ballCur = { x: PITCH_L / 2, y: PITCH_W / 2 };

    this._layout();
    this.app.ticker.add(() => this._frame());
  }

  _layout() {
    const w = this.app.renderer.width / this.app.renderer.resolution;
    const h = this.app.renderer.height / this.app.renderer.resolution;
    const aspect = PITCH_L / PITCH_W;
    let pw = w * 0.94, ph = pw / aspect;
    if (ph > h * 0.94) { ph = h * 0.94; pw = ph * aspect; }
    this.L = { w, h, pw, ph, ox: (w - pw) / 2, oy: (h - ph) / 2, scale: pw / PITCH_L };
  }

  mx(m) { return this.L.ox + (m / PITCH_L) * this.L.pw; }
  my(m) { return this.L.oy + (m / PITCH_W) * this.L.ph; }
  bx(b) { return this.L.ox + b * this.L.pw; }
  by(b) { return this.L.oy + b * this.L.ph; }
  boardFromScreen(px, py) {
    return { boardX: (px - this.L.ox) / this.L.pw, boardY: (py - this.L.oy) / this.L.ph };
  }

  _drawPitch() {
    const g = this.pitchLayer;
    const s = this.L.scale;
    g.clear();
    g.beginFill(COL_GRASS);
    g.drawRect(this.mx(0), this.my(0), this.L.pw, this.L.ph);
    g.endFill();

    g.lineStyle(Math.max(2, s * 0.12), COL_LINE, 0.85);
    g.drawRect(this.mx(0), this.my(0), this.L.pw, this.L.ph);          // touchlines
    g.moveTo(this.mx(52.5), this.my(0));
    g.lineTo(this.mx(52.5), this.my(68));                              // halfway
    g.drawCircle(this.mx(52.5), this.my(34), 9.15 * s);                // centre circle

    for (const near of [true, false]) {
      const px = near ? this.mx(0) : this.mx(105 - 16.5);
      g.drawRect(px, this.my(13.84), 16.5 * s, 40.32 * s);
      const gx = near ? this.mx(0) : this.mx(105 - 5.5);
      g.drawRect(gx, this.my(24.84), 5.5 * s, 18.32 * s);
    }

    g.beginFill(COL_LINE);
    g.drawCircle(this.mx(52.5), this.my(34), Math.max(2, s * 0.15));   // centre spot
    g.drawCircle(this.mx(11), this.my(34), Math.max(2, s * 0.15));     // penalty spots
    g.drawCircle(this.mx(94), this.my(34), Math.max(2, s * 0.15));
    g.endFill();
  }

  _drawOverlay() {
    const g = this.overlayLayer;
    g.clear();
    if (this.showCalibration && this.state?.calibrationOverlay) this._drawCalibration(g);
    // Shadows first: the offside line and the players read on top of them.
    if (this.state?.shadowOverlay) this._drawShadows(g);
    else if (this._shadowLabel) this._shadowLabel.visible = false;
    if (this.state?.offsideOverlay) this._drawOffside(g);
    else if (this._offsideLabels) this._offsideLabels.forEach((t) => (t.visible = false));
  }

  // Defender reach shadows. The server sends one polygon per defending player
  // (where they can get to within shadowSeconds, given their current velocity);
  // we just fill them. The fills are deliberately translucent so overlaps
  // compound into darker double-covered areas and the gaps between them stay
  // bare grass -- those gaps are the unmarked zones.
  _drawShadows(g) {
    const shadows = this.state.shadows || [];
    if (!shadows.length) return;
    for (const s of shadows) {
      if (!s.points || s.points.length < 3) continue;
      const colour = s.team === "home" ? COL_HOME : COL_AWAY;
      const flat = [];
      for (const [x, y] of s.points) flat.push(this.mx(x), this.my(y));
      g.lineStyle(Math.max(1, this.L.scale * 0.05), colour, 0.45);
      g.beginFill(colour, 0.15);
      g.drawPolygon(flat);
      g.endFill();
    }
    this._shadowText(shadows[0].team);
  }

  _shadowText(team) {
    if (!this._shadowLabel) {
      this._shadowLabel = new PIXI.Text("", { fontFamily: "system-ui, sans-serif",
                                              fontSize: 12, fill: 0xffffff, fontWeight: "bold" });
      this._shadowLabel.anchor.set(0, 1);
      this.overlayLayer.addChild(this._shadowLabel);
    }
    const t = this._shadowLabel;
    t.visible = true;
    t.style.fill = team === "home" ? COL_HOME : COL_AWAY;
    t.style.fontSize = Math.max(10, this.L.scale * 0.85);
    t.text = `REACH · ${(this.state.shadowSeconds ?? 2).toFixed(1)}s · ${team}`;
    t.position.set(this.mx(0), this.my(0) - 4);
  }

  _drawCalibration(g) {
    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const r = Math.max(18, this.L.scale * 1.6);
    corners.forEach(([cx, cy], i) => {
      const x = this.bx(cx), y = this.by(cy);
      g.lineStyle(4, COL_CALIB, 1);
      g.drawCircle(x, y, r);
      g.drawCircle(x, y, r * 0.45);
      g.moveTo(x - r * 1.6, y); g.lineTo(x + r * 1.6, y);
      g.moveTo(x, y - r * 1.6); g.lineTo(x, y + r * 1.6);
      this._corner(i, x, y, r);
    });
  }

  // Offside line = x of the second-last defender on each team, on the half they
  // defend. Which end each team defends is decided by team centroid so the line
  // stays correct after half-time (when the tracking flips direction).
  _drawOffside(g) {
    const xs = { home: [], away: [] };
    for (const p of this.state.players) (xs[p.team] || (xs[p.team] = [])).push(p.x);
    if (!xs.home || !xs.away || xs.home.length < 2 || xs.away.length < 2) return;
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const homeDefendsLeft = mean(xs.home) < mean(xs.away);
    xs.home.sort((a, b) => a - b);
    xs.away.sort((a, b) => a - b);
    const homeLine = homeDefendsLeft ? xs.home[1] : xs.home[xs.home.length - 2];
    const awayLine = homeDefendsLeft ? xs.away[xs.away.length - 2] : xs.away[1];
    this._offsideLine(g, homeLine, COL_HOME, "home", 0);
    this._offsideLine(g, awayLine, COL_AWAY, "away", 1);
  }

  _offsideLine(g, xMetres, colour, label, slot) {
    const x = this.mx(xMetres);
    const yTop = this.my(0), yBot = this.my(68);
    g.lineStyle(Math.max(2, this.L.scale * 0.14), colour, 0.85);
    const dash = Math.max(6, this.L.scale * 0.8);
    const gap = dash * 0.55;
    for (let y = yTop; y < yBot; y += dash + gap) {
      g.moveTo(x, y); g.lineTo(x, Math.min(y + dash, yBot));
    }
    if (!this._offsideLabels) {
      this._offsideLabels = [0, 1].map(() => {
        const t = new PIXI.Text("", { fontFamily: "system-ui, sans-serif",
                                      fontSize: 12, fill: 0xffffff, fontWeight: "bold" });
        t.anchor.set(0.5, 1);
        this.overlayLayer.addChild(t);
        return t;
      });
    }
    const t = this._offsideLabels[slot];
    t.visible = true;
    t.style.fill = colour;
    t.style.fontSize = Math.max(10, this.L.scale * 0.85);
    t.text = `OFFSIDE · ${label}`;
    t.position.set(x, yTop - 4);
  }

  _corner(i, x, y, r) {
    if (!this._cornerLabels) {
      this._cornerLabels = [0, 1, 2, 3].map(() => {
        const t = new PIXI.Text("", { fontFamily: "monospace", fontSize: 28,
                                      fill: COL_CALIB, fontWeight: "bold" });
        t.anchor.set(0.5);
        this.overlayLayer.addChild(t);
        return t;
      });
    }
    const t = this._cornerLabels[i];
    t.text = String(i + 1);
    t.visible = true;
    t.position.set(x + (i === 0 || i === 3 ? r * 2.2 : -r * 2.2),
                   y + (i < 2 ? r * 2.2 : -r * 2.2));
  }

  _frame() {
    const w = this.app.renderer.width / this.app.renderer.resolution;
    if (!this.L || Math.abs(w - this.L.w) > 1 ||
        Math.abs(this.app.renderer.height / this.app.renderer.resolution - this.L.h) > 1) {
      this._layout();
      this._drawPitch();
    }
    this.fps = this.app.ticker.FPS;
    this._drawOverlay();
    if (this.state) { this._frameplayers(); this._frameCursors(); }
    if (this._cornerLabels && !(this.showCalibration && this.state?.calibrationOverlay)) {
      this._cornerLabels.forEach((t) => (t.visible = false));
    }
  }

  _frameplayers() {
    const seen = new Set();
    for (const p of this.state.players) {
      seen.add(p.id);
      let sp = this.sprites.get(p.id);
      if (!sp) sp = this._makePlayer(p);
      const k = p.grabbed ? 0.55 : 0.22;
      sp.cur.x += (p.x - sp.cur.x) * k;
      sp.cur.y += (p.y - sp.cur.y) * k;

      const r = Math.max(8, this.L.scale * 1.15);
      const g = sp.gfx;
      g.clear();
      if (p.grabbed) { g.lineStyle(4, 0xffffff, 1); }
      else if (p.edited) { g.lineStyle(2.5, 0xffffff, 0.55); }
      else { g.lineStyle(1.5, 0x000000, 0.35); }
      g.beginFill(p.team === "home" ? COL_HOME : COL_AWAY, 1);
      g.drawCircle(0, 0, p.grabbed ? r * 1.2 : r);
      g.endFill();
      g.position.set(this.mx(sp.cur.x), this.my(sp.cur.y));
      sp.label.position.copyFrom(g.position);
      sp.label.style.fontSize = Math.max(9, r * 0.95);
    }
    for (const [id, sp] of this.sprites) {
      if (!seen.has(id)) { sp.gfx.destroy(); sp.label.destroy(); this.sprites.delete(id); }
    }

    const b = this.state.ball;
    this.ballCur.x += (b.x - this.ballCur.x) * 0.25;
    this.ballCur.y += (b.y - this.ballCur.y) * 0.25;
    this.ball.clear();
    this.ball.beginFill(COL_BALL);
    this.ball.drawCircle(this.mx(this.ballCur.x), this.my(this.ballCur.y),
                         Math.max(4, this.L.scale * 0.5));
    this.ball.endFill();
  }

  _makePlayer(p) {
    const gfx = new PIXI.Graphics();
    const label = new PIXI.Text(String(p.number), {
      fontFamily: "system-ui, sans-serif", fontSize: 14, fill: 0x061018, fontWeight: "bold",
    });
    label.anchor.set(0.5);
    this.playersLayer.addChild(gfx, label);
    const sp = { gfx, label, cur: { x: p.x, y: p.y } };
    this.sprites.set(p.id, sp);
    return sp;
  }

  _frameCursors() {
    const seen = new Set();
    for (const c of this.state.cursors || []) {
      seen.add(c.handId);
      let g = this.cursorSprites.get(c.handId);
      if (!g) { g = new PIXI.Graphics(); this.cursorsLayer.addChild(g); this.cursorSprites.set(c.handId, g); }
      const x = this.bx(c.boardX), y = this.by(c.boardY);
      const r = Math.max(10, this.L.scale * 0.9);
      g.clear();
      if (c.grabbing) {
        g.beginFill(0xffffff, 0.85);
        g.drawCircle(x, y, r * 0.55);
        g.endFill();
      }
      g.lineStyle(3, 0xffffff, c.grabbing ? 1 : 0.6);
      g.drawCircle(x, y, r);
      g.moveTo(x - r * 1.5, y); g.lineTo(x - r * 0.7, y);
      g.moveTo(x + r * 0.7, y); g.lineTo(x + r * 1.5, y);
      g.moveTo(x, y - r * 1.5); g.lineTo(x, y - r * 0.7);
      g.moveTo(x, y + r * 0.7); g.lineTo(x, y + r * 1.5);
    }
    for (const [id, g] of this.cursorSprites) {
      if (!seen.has(id)) { g.destroy(); this.cursorSprites.delete(id); }
    }
  }

  applyState(state) { this.state = state; }

  hitTest(boardX, boardY) {
    if (!this.state) return null;
    const x = boardX * PITCH_L, y = boardY * PITCH_W;
    let best = null, bestD2 = 3.0 * 3.0;
    for (const p of this.state.players) {
      const d2 = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d2 < bestD2) { best = p; bestD2 = d2; }
    }
    return best;
  }

  enableMouse() {
    const view = this.app.view;
    let dragging = null;
    const board = (e) => {
      const rect = view.getBoundingClientRect();
      return this.boardFromScreen(e.clientX - rect.left, e.clientY - rect.top);
    };
    view.addEventListener("pointerdown", (e) => {
      const { boardX, boardY } = board(e);
      const p = this.hitTest(boardX, boardY);
      if (!p) return;
      dragging = p.id;
      view.setPointerCapture(e.pointerId);
      this.onDragStart(p.id, boardX, boardY);
    });
    view.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const { boardX, boardY } = board(e);
      this.onDragMove(dragging, boardX, boardY);
    });
    const end = () => { if (dragging) { this.onDragEnd(dragging); dragging = null; } };
    view.addEventListener("pointerup", end);
    view.addEventListener("pointercancel", end);
    view.style.touchAction = "none";
    view.style.cursor = "crosshair";
  }
}
