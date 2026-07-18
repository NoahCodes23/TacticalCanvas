const PITCH_L = 105;
const PITCH_W = 68;

const COL_GRASS = 0x0d3b1a;   
const COL_LINE = 0xe8f5e9;
const COL_HOME = 0x38bdf8;
const COL_AWAY = 0xfb7185;
const COL_BALL = 0xfde047;
const COL_CALIB = 0xff00ff;   
const COL_AI_PASS = [0xfde047, 0x4ade80, 0xc084fc];
const COL_AI_FAINT = 0x94a3b8;
const COL_AI_TARGET = 0x22d3ee;

export class PitchRenderer {
  constructor(el, { showCalibration = true, quality = "sharp" } = {}) {
    this.el = el;
    this.showCalibration = showCalibration;
    this.state = null;
    this.fps = 0;

    const params = new URLSearchParams(window.location.search);
    const requestedQuality = params.get("quality");
    this.quality = requestedQuality === "performance" || requestedQuality === "sharp"
      ? requestedQuality : quality === "performance" ? "performance" : "sharp";
    const requestedScale = Number(params.get("renderScale"));
    this.renderScaleOverride = Number.isFinite(requestedScale) && requestedScale > 0
      ? Math.min(2, Math.max(1, requestedScale)) : null;
    this.renderResolution = this._desiredResolution();
    // Pixi Text is rasterized into its own texture. Keeping those textures at
    // least 2x prevents small glyphs becoming blocky even in performance mode.
    this.textResolution = Math.max(2, this.renderResolution);
    this.textObjects = new Set();

    this.onDragStart = () => {};
    this.onDragMove = () => {};
    this.onDragEnd = () => {};

    this.app = new PIXI.Application({
      resizeTo: el,
      backgroundColor: 0x000000,
      antialias: this.quality !== "performance",
      resolution: this.renderResolution,
      autoDensity: true,
      powerPreference: "high-performance",
    });
    el.appendChild(this.app.view);

    this.pitchLayer = new PIXI.Graphics();
    this.pitchControlLayer = new PIXI.Container();  // Voronoi shading, under everything
    this.overlayLayer = new PIXI.Graphics();
    this.playersLayer = new PIXI.Container();
    this.cursorsLayer = new PIXI.Container();
    this.app.stage.addChild(this.pitchLayer, this.pitchControlLayer, this.overlayLayer,
                            this.playersLayer, this.cursorsLayer);
    this._initPitchControl();

    this.sprites = new Map();  
    this.cursorSprites = new Map();
    this.ball = new PIXI.Graphics();
    this.playersLayer.addChild(this.ball);
    this.ballCur = { x: PITCH_L / 2, y: PITCH_W / 2 };
    this.geometryVersion = 0;
    this.overlayDirty = true;

    this._layout();
    this._drawPitch();
    this._drawBallGeometry();
    this.app.ticker.add(() => this._frame());
  }

  _desiredResolution() {
    if (this.renderScaleOverride != null) return this.renderScaleOverride;
    if (this.quality === "performance") return 1;
    // Supersample even on a 1x display, but cap at 2x so a 1080p projector
    // never asks the GPU to render beyond a 4K backing buffer.
    return Math.min(2, Math.max(1.5, window.devicePixelRatio || 1));
  }

  _makeText(value, style) {
    const text = new PIXI.Text(value, style);
    text.resolution = this.textResolution;
    this.textObjects.add(text);
    return text;
  }

  _syncDisplayResolution() {
    const desired = this._desiredResolution();
    if (Math.abs(desired - this.renderResolution) < 0.01) return;
    this.renderResolution = desired;
    this.textResolution = Math.max(2, desired);
    this.app.renderer.resolution = desired;
    this.app.renderer.resize(this.el.clientWidth, this.el.clientHeight);
    for (const text of [...this.textObjects]) {
      if (text.destroyed) this.textObjects.delete(text);
      else text.resolution = this.textResolution;
    }
  }

  _layout() {
    const w = this.app.renderer.width / this.app.renderer.resolution;
    const h = this.app.renderer.height / this.app.renderer.resolution;
    const aspect = PITCH_L / PITCH_W;
    let pw = w * 0.94, ph = pw / aspect;
    if (ph > h * 0.94) { ph = h * 0.94; pw = ph * aspect; }
    this.L = { w, h, pw, ph, ox: (w - pw) / 2, oy: (h - ph) / 2, scale: pw / PITCH_L };
    this.geometryVersion++;
    this.overlayDirty = true;
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
    this.overlayDirty = false;
    const g = this.overlayLayer;
    g.clear();
    if (this.showCalibration && this.state?.calibrationOverlay) this._drawCalibration(g);
    // Shadows first: the offside line and the players read on top of them.
    if (this.state?.shadowOverlay) this._drawShadows(g);
    else if (this._shadowLabel) this._shadowLabel.visible = false;
    if (this.state?.experiments?.receiverTargets) this._drawAiTargets(g);
    else this._hideTextPool(this._aiTargetLabels);
    if (this.state?.experiments?.passRecommendations) this._drawAiPasses(g);
    else {
      this._hideTextPool(this._aiPassLabels);
      this._hideTextPool(this._aiCarrierLabels);
    }
    if (this.state?.offsideOverlay) this._drawOffside(g);
    else if (this._offsideLabels) this._offsideLabels.forEach((t) => (t.visible = false));
    if (this.state?.formationOverlay) this._drawFormation();
    else if (this._formationLabels) this._formationLabels.forEach((t) => (t.visible = false));
  }

  _hideTextPool(pool) {
    if (pool) pool.forEach((text) => (text.visible = false));
  }

  _textFromPool(property, index, style) {
    if (!this[property]) this[property] = [];
    while (this[property].length <= index) {
      const text = this._makeText("", style);
      text.anchor.set(0.5);
      this.overlayLayer.addChild(text);
      this[property].push(text);
    }
    return this[property][index];
  }

  // Every candidate is shown faintly; the top three are ranked, coloured and
  // labelled with completion probability and expected-value score.
  _drawAiPasses(g) {
    const passes = this.state?.experimentalAnalysis?.passes || [];
    this._hideTextPool(this._aiPassLabels);
    this._hideTextPool(this._aiCarrierLabels);
    if (!passes.length) return;

    const origin = passes[0].from;
    if (origin) {
      const carrierX = this.mx(origin.x), carrierY = this.my(origin.y);
      const radius = Math.max(11, this.L.scale * 1.45);
      g.lineStyle(Math.max(2.5, this.L.scale * 0.2), COL_AI_PASS[0], 0.95);
      g.drawCircle(carrierX, carrierY, radius);
      const carrierLabel = this._textFromPool("_aiCarrierLabels", 0, {
        fontFamily: "ui-monospace, monospace", fontSize: 14,
        fill: COL_AI_PASS[0], fontWeight: "bold", stroke: 0x071018, strokeThickness: 3,
      });
      carrierLabel.visible = true;
      carrierLabel.style.fontSize = Math.max(14, this.L.scale * 1.0);
      carrierLabel.text = `BALL CARRIER #${this.state.experimentalAnalysis?.context?.ballCarrierNumber ?? "?"}`;
      carrierLabel.position.set(carrierX, carrierY + radius * 1.5);
    }

    // Draw low-ranked alternatives first so the three recommendations stay
    // visually on top where several paths share the same opening segment.
    const ordered = [...passes].sort((a, b) => Number(a.recommended) - Number(b.recommended));
    for (const pass of ordered) {
      const start = pass.from, end = pass.to;
      if (!start || !end) continue;
      const recommended = pass.rank <= 3;
      const colour = recommended ? COL_AI_PASS[pass.rank - 1] : COL_AI_FAINT;
      const alpha = recommended ? 0.92 : 0.16;
      const width = recommended ? Math.max(2, this.L.scale * 0.18) : 1;
      const x1 = this.mx(start.x), y1 = this.my(start.y);
      const x2 = this.mx(end.x), y2 = this.my(end.y);
      g.lineStyle(width, colour, alpha);
      g.moveTo(x1, y1); g.lineTo(x2, y2);

      const angle = Math.atan2(y2 - y1, x2 - x1);
      const head = Math.max(7, this.L.scale * (recommended ? 0.8 : 0.5));
      g.moveTo(x2, y2);
      g.lineTo(x2 - head * Math.cos(angle - 0.48), y2 - head * Math.sin(angle - 0.48));
      g.moveTo(x2, y2);
      g.lineTo(x2 - head * Math.cos(angle + 0.48), y2 - head * Math.sin(angle + 0.48));

      if (recommended) {
        const label = this._textFromPool("_aiPassLabels", pass.rank - 1, {
          fontFamily: "ui-monospace, monospace", fontSize: 14,
          fill: colour, fontWeight: "bold", stroke: 0x071018, strokeThickness: 3,
        });
        label.visible = true;
        label.style.fill = colour;
        label.style.fontSize = Math.max(14, this.L.scale * 1.0);
        const probability = Math.round((pass.completionProbability || 0) * 100);
        const signedScore = pass.score >= 0 ? `+${pass.score}` : String(pass.score);
        label.text = `#${pass.rank}  P ${probability}%  EV ${signedScore}`;
        label.position.set((x1 + x2) / 2, (y1 + y2) / 2 - Math.max(12, this.L.scale * 1.0));
      }
    }
  }

  // Suggested off-ball moves for the current top receivers. These are local
  // reachable-position searches, kept visually distinct from actual passes.
  _drawAiTargets(g) {
    const targets = this.state?.experimentalAnalysis?.receiverTargets || [];
    this._hideTextPool(this._aiTargetLabels);
    targets.forEach((target, index) => {
      const start = target.from, end = target.to;
      if (!start || !end) return;
      const x1 = this.mx(start.x), y1 = this.my(start.y);
      const x2 = this.mx(end.x), y2 = this.my(end.y);
      const dx = x2 - x1, dy = y2 - y1;
      const length = Math.hypot(dx, dy);
      if (length < 1) return;
      const ux = dx / length, uy = dy / length;
      const dash = Math.max(5, this.L.scale * 0.55);
      g.lineStyle(Math.max(1.5, this.L.scale * 0.11), COL_AI_TARGET, 0.78);
      for (let travelled = 0; travelled < length; travelled += dash * 1.7) {
        const finish = Math.min(travelled + dash, length);
        g.moveTo(x1 + ux * travelled, y1 + uy * travelled);
        g.lineTo(x1 + ux * finish, y1 + uy * finish);
      }
      const radius = Math.max(7, this.L.scale * 0.75);
      g.drawCircle(x2, y2, radius);
      g.moveTo(x2 - radius * 1.35, y2); g.lineTo(x2 + radius * 1.35, y2);
      g.moveTo(x2, y2 - radius * 1.35); g.lineTo(x2, y2 + radius * 1.35);

      const label = this._textFromPool("_aiTargetLabels", index, {
        fontFamily: "ui-monospace, monospace", fontSize: 13,
        fill: COL_AI_TARGET, fontWeight: "bold", stroke: 0x071018, strokeThickness: 3,
      });
      label.visible = true;
      label.style.fontSize = Math.max(13, this.L.scale * 0.95);
      label.text = `#${target.playerNumber} move ${target.moveDistanceM}m  +${target.improvement} EV`;
      label.position.set(x2, y2 - radius * 1.8);
    });
  }

  // Auto-detected formation labels, one per team, above the top touchline.
  // The strings come from the server; here we just place them.
  _drawFormation() {
    if (!this._formationLabels) {
      this._formationLabels = [0, 1].map(() => {
        const t = this._makeText("", { fontFamily: "system-ui, sans-serif",
                                       fontSize: 14, fill: 0xffffff, fontWeight: "bold" });
        this.overlayLayer.addChild(t);
        return t;
      });
    }
    const [hLab, aLab] = this._formationLabels;
    const f = this.state.formations || {};
    const yTop = this.my(0) - 4;
    const fs = Math.max(14, this.L.scale * 1.05);

    hLab.anchor.set(0, 1);
    hLab.style.fill = COL_HOME;
    hLab.style.fontSize = fs;
    hLab.text = f.home ? `HOME · ${f.home}` : "HOME · —";
    hLab.position.set(this.mx(0), yTop);
    hLab.visible = true;

    aLab.anchor.set(1, 1);
    aLab.style.fill = COL_AWAY;
    aLab.style.fontSize = fs;
    aLab.text = f.away ? `${f.away} · AWAY` : "— · AWAY";
    aLab.position.set(this.mx(105), yTop);
    aLab.visible = true;
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
      this._shadowLabel = this._makeText("", { fontFamily: "system-ui, sans-serif",
                                                fontSize: 13, fill: 0xffffff, fontWeight: "bold" });
      this._shadowLabel.anchor.set(0, 1);
      this.overlayLayer.addChild(this._shadowLabel);
    }
    const t = this._shadowLabel;
    t.visible = true;
    t.style.fill = team === "home" ? COL_HOME : COL_AWAY;
    t.style.fontSize = Math.max(13, this.L.scale * 0.95);
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
  // Clamped to halfway: an attacker can't be offside in their own half, so if
  // the second-last defender has pushed past midfield the line stays at 52.5
  // rather than following them into the attacker's side.
  _drawOffside(g) {
    const xs = { home: [], away: [] };
    for (const p of this.state.players) (xs[p.team] || (xs[p.team] = [])).push(p.x);
    if (!xs.home || !xs.away || xs.home.length < 2 || xs.away.length < 2) return;
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const homeDefendsLeft = mean(xs.home) < mean(xs.away);
    xs.home.sort((a, b) => a - b);
    xs.away.sort((a, b) => a - b);
    const rawHome = homeDefendsLeft ? xs.home[1] : xs.home[xs.home.length - 2];
    const rawAway = homeDefendsLeft ? xs.away[xs.away.length - 2] : xs.away[1];
    const homeLine = homeDefendsLeft ? Math.min(rawHome, 52.5) : Math.max(rawHome, 52.5);
    const awayLine = homeDefendsLeft ? Math.max(rawAway, 52.5) : Math.min(rawAway, 52.5);
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
        const t = this._makeText("", { fontFamily: "system-ui, sans-serif",
                                       fontSize: 13, fill: 0xffffff, fontWeight: "bold" });
        t.anchor.set(0.5, 1);
        this.overlayLayer.addChild(t);
        return t;
      });
    }
    const t = this._offsideLabels[slot];
    t.visible = true;
    t.style.fill = colour;
    t.style.fontSize = Math.max(13, this.L.scale * 0.95);
    t.text = `OFFSIDE · ${label}`;
    t.position.set(x, yTop - 4);
  }

  _corner(i, x, y, r) {
    if (!this._cornerLabels) {
      this._cornerLabels = [0, 1, 2, 3].map(() => {
        const t = this._makeText("", { fontFamily: "monospace", fontSize: 28,
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
    this._syncDisplayResolution();
    const w = this.app.renderer.width / this.app.renderer.resolution;
    if (!this.L || Math.abs(w - this.L.w) > 1 ||
        Math.abs(this.app.renderer.height / this.app.renderer.resolution - this.L.h) > 1) {
      this._layout();
      this._drawPitch();
      this._drawBallGeometry();
    }
    this.fps = this.app.ticker.FPS;
    if (this.overlayDirty) this._drawOverlay();
    this._updatePitchControl();
    if (this.state) { this._frameplayers(); this._frameCursors(); }
    if (this._cornerLabels && !(this.showCalibration && this.state?.calibrationOverlay)) {
      this._cornerLabels.forEach((t) => (t.visible = false));
    }
  }

  // Voronoi pitch control: each cell of a low-res grid takes on the team colour
  // of the nearest player. Rendered as a small canvas texture stretched over the
  // pitch (LINEAR-filtered so the region edges look like soft shading rather
  // than pixel steps). Recomputed only when the server bumps state.revision,
  // which happens on every drag move -- so the shading is live under the coach's
  // finger without the renderer having to run the algorithm at 60 Hz.
  _initPitchControl() {
    const PC_W = 50, PC_H = 32;
    this._pcW = PC_W; this._pcH = PC_H;
    this._pcCanvas = document.createElement("canvas");
    this._pcCanvas.width = PC_W;
    this._pcCanvas.height = PC_H;
    this._pcCtx = this._pcCanvas.getContext("2d");
    this._pcImage = this._pcCtx.createImageData(PC_W, PC_H);
    const tex = PIXI.Texture.from(this._pcCanvas);
    tex.baseTexture.scaleMode = PIXI.SCALE_MODES.LINEAR;
    this._pcSprite = new PIXI.Sprite(tex);
    this._pcSprite.alpha = 0.42;
    this._pcSprite.visible = false;
    this.pitchControlLayer.addChild(this._pcSprite);
    this._pcLastRev = -1;
  }

  _updatePitchControl() {
    const on = this.state?.pitchControlOverlay && this.state.players?.length;
    if (!on) {
      this._pcSprite.visible = false;
      this._pcLastRev = -1;   // force redraw on re-enable
      return;
    }
    this._pcSprite.visible = true;
    this._pcSprite.position.set(this.mx(0), this.my(0));
    this._pcSprite.width = this.L.pw;
    this._pcSprite.height = this.L.ph;

    if (this._pcLastRev === this.state.revision) return;
    this._pcLastRev = this.state.revision;

    const w = this._pcW, h = this._pcH;
    const data = this._pcImage.data;
    const players = this.state.players;
    const sx = PITCH_L / w, sy = PITCH_W / h;
    // Split-channel constants for the two team colours (matches COL_HOME/AWAY).
    const HR = 0x38, HG = 0xbd, HB = 0xf8;
    const AR = 0xfb, AG = 0x71, AB = 0x85;
    for (let j = 0; j < h; j++) {
      const yM = (j + 0.5) * sy;
      for (let i = 0; i < w; i++) {
        const xM = (i + 0.5) * sx;
        let bestD2 = Infinity, home = true;
        for (const p of players) {
          const dx = p.x - xM, dy = p.y - yM;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) { bestD2 = d2; home = p.team === "home"; }
        }
        const idx = (j * w + i) * 4;
        if (home) { data[idx] = HR; data[idx + 1] = HG; data[idx + 2] = HB; }
        else      { data[idx] = AR; data[idx + 1] = AG; data[idx + 2] = AB; }
        data[idx + 3] = 255;
      }
    }
    this._pcCtx.putImageData(this._pcImage, 0, 0);
    this._pcSprite.texture.update();
  }

  _frameplayers() {
    const seen = new Set();
    for (const p of this.state.players) {
      seen.add(p.id);
      let sp = this.sprites.get(p.id);
      if (!sp) sp = this._makePlayer(p);
      if (p.grabbed) {
        // Hand coordinates were already filtered once in the vision worker.
        sp.cur.x = p.x;
        sp.cur.y = p.y;
      } else {
        sp.cur.x += (p.x - sp.cur.x) * 0.35;
        sp.cur.y += (p.y - sp.cur.y) * 0.35;
      }

      const styleKey = `${this.geometryVersion}:${p.team}:${p.grabbed}:${p.edited}`;
      if (sp.styleKey !== styleKey) {
        this._drawPlayerGeometry(sp, p);
        sp.styleKey = styleKey;
      }
      sp.gfx.position.set(this.mx(sp.cur.x), this.my(sp.cur.y));
      sp.label.position.copyFrom(sp.gfx.position);
    }
    for (const [id, sp] of this.sprites) {
      if (!seen.has(id)) {
        sp.gfx.destroy();
        this.textObjects.delete(sp.label);
        sp.label.destroy();
        this.sprites.delete(id);
      }
    }

    const b = this.state.ball;
    this.ballCur.x += (b.x - this.ballCur.x) * 0.25;
    this.ballCur.y += (b.y - this.ballCur.y) * 0.25;
    this.ball.position.set(this.mx(this.ballCur.x), this.my(this.ballCur.y));
  }

  _drawPlayerGeometry(sp, p) {
    const r = Math.max(8, this.L.scale * 1.15);
    const g = sp.gfx;
    g.clear();
    if (p.grabbed) { g.lineStyle(4, 0xffffff, 1); }
    else if (p.edited) { g.lineStyle(2.5, 0xffffff, 0.55); }
    else { g.lineStyle(1.5, 0x000000, 0.35); }
    g.beginFill(p.team === "home" ? COL_HOME : COL_AWAY, 1);
    g.drawCircle(0, 0, p.grabbed ? r * 1.2 : r);
    g.endFill();
    sp.label.style.fontSize = Math.max(12, r * 0.95);
  }

  _drawBallGeometry() {
    this.ball.clear();
    this.ball.beginFill(COL_BALL);
    this.ball.drawCircle(0, 0, Math.max(4, this.L.scale * 0.5));
    this.ball.endFill();
  }

  _makePlayer(p) {
    const gfx = new PIXI.Graphics();
    const label = this._makeText(String(p.number), {
      fontFamily: "system-ui, sans-serif", fontSize: 14, fill: 0x061018, fontWeight: "bold",
    });
    label.anchor.set(0.5);
    this.playersLayer.addChild(gfx, label);
    const sp = { gfx, label, cur: { x: p.x, y: p.y }, styleKey: "" };
    this.sprites.set(p.id, sp);
    return sp;
  }

  _frameCursors() {
    const seen = new Set();
    for (const c of this.state.cursors || []) {
      seen.add(c.handId);
      let g = this.cursorSprites.get(c.handId);
      if (!g) {
        g = new PIXI.Graphics();
        g.styleKey = "";
        this.cursorsLayer.addChild(g);
        this.cursorSprites.set(c.handId, g);
      }
      const x = this.bx(c.boardX), y = this.by(c.boardY);
      const r = Math.max(10, this.L.scale * 0.9);
      const styleKey = `${this.geometryVersion}:${c.grabbing}`;
      if (g.styleKey !== styleKey) {
        g.clear();
        if (c.grabbing) {
          g.beginFill(0xffffff, 0.85);
          g.drawCircle(0, 0, r * 0.55);
          g.endFill();
        }
        g.lineStyle(3, 0xffffff, c.grabbing ? 1 : 0.6);
        g.drawCircle(0, 0, r);
        g.moveTo(-r * 1.5, 0); g.lineTo(-r * 0.7, 0);
        g.moveTo(r * 0.7, 0); g.lineTo(r * 1.5, 0);
        g.moveTo(0, -r * 1.5); g.lineTo(0, -r * 0.7);
        g.moveTo(0, r * 0.7); g.lineTo(0, r * 1.5);
        g.styleKey = styleKey;
      }
      g.position.set(x, y);
    }
    for (const [id, g] of this.cursorSprites) {
      if (!seen.has(id)) { g.destroy(); this.cursorSprites.delete(id); }
    }
  }

  applyState(state) {
    const aiOverlay = state.experiments?.passRecommendations || state.experiments?.receiverTargets;
    const overlaysActive = state.calibrationOverlay || state.shadowOverlay
      || state.offsideOverlay || state.formationOverlay || aiOverlay;
    if (overlaysActive
        || this.state?.calibrationOverlay !== state.calibrationOverlay
        || this.state?.shadowOverlay !== state.shadowOverlay
        || this.state?.offsideOverlay !== state.offsideOverlay
        || this.state?.formationOverlay !== state.formationOverlay
        || this.state?.experiments?.passRecommendations !== state.experiments?.passRecommendations
        || this.state?.experiments?.receiverTargets !== state.experiments?.receiverTargets) {
      this.overlayDirty = true;
    }
    this.state = state;
  }

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
