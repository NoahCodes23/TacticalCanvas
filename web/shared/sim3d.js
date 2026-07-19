// 3D pop-out pitch for the simulation. Two implementations behind one facade:
//
// * `GLSimView` — the real thing: a Three.js WebGL scene (vendored module, no
//   build step) with an orbit camera, soft shadows, billboarded shirt numbers,
//   and a glowing pass lane. This is the demo centrepiece.
// * `CssSim3DView` — the original CSS-3D tilted plane, kept verbatim as the
//   never-breaks fallback. If the vendored module fails to load or WebGL
//   context creation throws (projector GPU, remote desktop, ancient flags),
//   the facade drops to it silently and the demo still runs.
//
// Both are pure *readers* of the same authoritative state the 2D pitch uses
// (state.players, state.ball, state.simulation) — they own no game logic.
// Contract: new Sim3DView(mountEl) · applyState(state) · show() · hide().
// The view shows itself when state.simulation.active and hides otherwise.

const HOME = "#38bdf8";
const AWAY = "#fb7185";
const BALL = "#fde047";
const LERP = 0.28;           // position smoothing per frame

export class Sim3DView {
  constructor(mount) {
    this.impl = null;
    this._pending = null;
    this._init(mount);
  }

  async _init(mount) {
    try {
      const THREE = await import("/static/vendor/three.module.min.js");
      this.impl = new GLSimView(mount, THREE);
    } catch (err) {
      console.warn("[sim3d] WebGL scene unavailable, using CSS fallback:", err);
      this.impl = new CssSim3DView(mount);
    }
    if (this._pending) this.impl.applyState(this._pending);
  }

  applyState(state) {
    if (this.impl) this.impl.applyState(state);
    else this._pending = state;   // arrives before the async import settles
  }

  show() { this.impl?.show(); }
  hide() { this.impl?.hide(); }
}

/* ------------------------------------------------------------------------- *
 * WebGL implementation
 *
 * World space is pitch metres, y-up, origin at the pitch centre spot:
 * world.x = pitch.x - length/2, world.z = pitch.y - width/2. The camera
 * orbits the centre; the default view is from the near touchline.
 * ------------------------------------------------------------------------- */
class GLSimView {
  constructor(mount, THREE) {
    this.T = THREE;
    this.mount = mount;
    this.pitchL = 105;
    this.pitchW = 68;
    this.state = null;
    this.visible = false;
    this.tokens = new Map();       // playerId -> token
    this.ballCur = { x: 52.5, y: 34 };
    this._spriteCache = new Map();

    // Throws on machines with no WebGL — caught by the facade.
    this.renderer = new THREE.WebGLRenderer({
      antialias: true, alpha: true, powerPreference: "high-performance",
    });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.root = document.createElement("div");
    this.root.className = "sim3d-root";
    this.root.style.cssText =
      "position:absolute;inset:0;display:none;z-index:40;overflow:hidden;" +
      "background:radial-gradient(130% 130% at 50% 0%,#0b1a12 0%,#06100b 60%,#020403 100%);" +
      "cursor:grab;";
    this.renderer.domElement.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
    this.root.appendChild(this.renderer.domElement);
    mount.appendChild(this.root);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.5, 600);

    this._buildLights();
    this._buildPitch();
    this._buildBall();
    this._buildLane();
    this._buildCarrierRing();
    this._initOrbit();
    this._buildTooltip();

    this.raycaster = new THREE.Raycaster();
    this._mouse = { x: 0, y: 0, px: 0, py: 0, over: false };
    this._hoverId = null;

    this._raf = requestAnimationFrame(() => this._frame());
  }

  // -- static scene ---------------------------------------------------------
  _buildLights() {
    const T = this.T;
    this.scene.add(new T.HemisphereLight(0xbfe8d2, 0x06110b, 1.15));
    const sun = new T.DirectionalLight(0xffffff, 2.4);
    sun.position.set(-45, 80, -35);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const cam = sun.shadow.camera;
    cam.left = -75; cam.right = 75; cam.top = 55; cam.bottom = -55;
    cam.near = 10; cam.far = 220;
    sun.shadow.bias = -0.0004;
    this.scene.add(sun);
  }

  _pitchTexture() {
    // The pitch surface is one big canvas texture: stripes plus markings,
    // drawn at 16 px/m so lines stay crisp under the closest zoom.
    const T = this.T;
    const S = 16;
    const c = document.createElement("canvas");
    c.width = this.pitchL * S; c.height = this.pitchW * S;
    const g = c.getContext("2d");
    const stripe = this.pitchL / 10;
    for (let i = 0; i < 10; i++) {
      g.fillStyle = i % 2 ? "#12492a" : "#0f4023";
      g.fillRect(i * stripe * S, 0, stripe * S + 1, c.height);
    }
    g.strokeStyle = "rgba(220,239,228,0.78)";
    g.lineWidth = 0.14 * S;
    const box = (x, y, w, h) => g.strokeRect(x * S, y * S, w * S, h * S);
    const W = this.pitchW, L = this.pitchL;
    box(0.3, 0.3, L - 0.6, W - 0.6);                       // touchlines
    g.beginPath(); g.moveTo(L / 2 * S, 0.3 * S); g.lineTo(L / 2 * S, (W - 0.3) * S); g.stroke();
    g.beginPath(); g.arc(L / 2 * S, W / 2 * S, 9.15 * S, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.arc(L / 2 * S, W / 2 * S, 0.3 * S, 0, Math.PI * 2);
    g.fillStyle = "rgba(220,239,228,0.78)"; g.fill();
    box(0.3, W / 2 - 20.16, 16.5, 40.32);                  // penalty areas
    box(L - 16.8, W / 2 - 20.16, 16.5, 40.32);
    box(0.3, W / 2 - 9.16, 5.5, 18.32);                    // six-yard boxes
    box(L - 5.8, W / 2 - 9.16, 5.5, 18.32);
    const tex = new T.CanvasTexture(c);
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  _buildPitch() {
    const T = this.T;
    // Dark apron under everything, so shadows fade into the void gracefully.
    const apron = new T.Mesh(
      new T.PlaneGeometry(420, 420),
      new T.MeshLambertMaterial({ color: 0x03130b }));
    apron.rotation.x = -Math.PI / 2;
    apron.position.y = -0.06;
    this.scene.add(apron);

    this.pitchMesh = new T.Mesh(
      new T.PlaneGeometry(this.pitchL + 2, this.pitchW + 2),
      new T.MeshLambertMaterial({ map: this._pitchTexture() }));
    this.pitchMesh.rotation.x = -Math.PI / 2;
    this.pitchMesh.receiveShadow = true;
    this.scene.add(this.pitchMesh);

    // Goal frames: posts + crossbar, 7.32 x 2.44 m.
    const mat = new T.MeshStandardMaterial({ color: 0xf1f5f4, roughness: 0.4 });
    const half = this.pitchL / 2;
    for (const dir of [-1, 1]) {
      const goal = new T.Group();
      const post = () => new T.Mesh(new T.BoxGeometry(0.12, 2.44, 0.12), mat);
      const p1 = post(); p1.position.set(0, 1.22, -3.66);
      const p2 = post(); p2.position.set(0, 1.22, 3.66);
      const bar = new T.Mesh(new T.BoxGeometry(0.12, 0.12, 7.44), mat);
      bar.position.set(0, 2.44, 0);
      for (const m of [p1, p2, bar]) m.castShadow = true;
      goal.add(p1, p2, bar);
      goal.position.set(dir * half, 0, 0);
      this.scene.add(goal);
    }
  }

  _buildBall() {
    const T = this.T;
    this.ballMesh = new T.Mesh(
      new T.SphereGeometry(0.34, 24, 16),
      new T.MeshStandardMaterial({
        color: BALL, emissive: 0x8a7a10, roughness: 0.35,
      }));
    this.ballMesh.castShadow = true;
    this.scene.add(this.ballMesh);
  }

  _buildLane() {
    // The active pass, as an additive glowing strip laid on the grass.
    const T = this.T;
    const c = document.createElement("canvas");
    c.width = 256; c.height = 16;
    const g = c.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 256, 0);
    grad.addColorStop(0, "rgba(255,255,255,0)");
    grad.addColorStop(1, "rgba(255,255,255,0.95)");
    g.fillStyle = grad; g.fillRect(0, 0, 256, 16);
    this.laneTex = new T.CanvasTexture(c);
    this.lane = new T.Mesh(
      new T.PlaneGeometry(1, 1),
      new T.MeshBasicMaterial({
        map: this.laneTex, transparent: true, opacity: 0,
        blending: T.AdditiveBlending, depthWrite: false, color: HOME,
      }));
    this.lane.rotation.x = -Math.PI / 2;
    this.lane.position.y = 0.04;
    this.scene.add(this.lane);
  }

  _buildCarrierRing() {
    const T = this.T;
    this.ring = new T.Mesh(
      new T.RingGeometry(1.15, 1.6, 40),
      new T.MeshBasicMaterial({
        color: BALL, transparent: true, opacity: 0.85, depthWrite: false,
      }));
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.position.y = 0.05;
    this.ring.visible = false;
    this.scene.add(this.ring);
  }

  // -- players --------------------------------------------------------------
  _numberSprite(number, color) {
    const key = `${number}|${color}`;
    let tex = this._spriteCache.get(key);
    if (!tex) {
      const c = document.createElement("canvas");
      c.width = c.height = 128;
      const g = c.getContext("2d");
      g.beginPath(); g.arc(64, 64, 56, 0, Math.PI * 2);
      g.fillStyle = color; g.fill();
      g.lineWidth = 6; g.strokeStyle = "rgba(4,18,26,0.85)"; g.stroke();
      g.fillStyle = "#04121a";
      g.font = "700 56px ui-monospace, monospace";
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(String(number), 64, 68);
      tex = new this.T.CanvasTexture(c);
      tex.colorSpace = this.T.SRGBColorSpace;
      this._spriteCache.set(key, tex);
    }
    const sprite = new this.T.Sprite(new this.T.SpriteMaterial({ map: tex }));
    sprite.scale.set(1.45, 1.45, 1);
    sprite.position.y = 2.85;
    return sprite;
  }

  _token(p) {
    let t = this.tokens.get(p.id);
    if (t) return t;
    const T = this.T;
    const group = new T.Group();
    const body = new T.Mesh(
      new T.CylinderGeometry(0.82, 0.95, 1.8, 24),
      new T.MeshStandardMaterial({ roughness: 0.55, metalness: 0.08 }));
    body.position.y = 0.9;
    body.castShadow = true;
    body.userData.playerId = p.id;
    group.add(body);
    group.add(this._numberSprite(p.number, p.team === "home" ? HOME : AWAY));
    this.scene.add(group);
    t = { group, body, sprite: group.children[1], cur: { x: p.x, y: p.y }, key: "" };
    this.tokens.set(p.id, t);
    return t;
  }

  // -- orbit camera ---------------------------------------------------------
  _initOrbit() {
    // yaw: around the pitch; el: elevation above the grass; dist: dolly.
    // Goal values are eased toward every frame, which gives free damping and
    // makes the show() intro a plain goal-state change.
    this.orbit = { yaw: 0, el: 0.72, dist: 74 };
    this._orbitGoal = { yaw: 0, el: 0.72, dist: 74 };
    this._defaultView = { yaw: 0, el: 0.72, dist: 74 };

    const el = this.root;
    this._dragging = false;
    let lastX = 0, lastY = 0;
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this._dragging = true; lastX = e.clientX; lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
      el.style.cursor = "grabbing";
    });
    el.addEventListener("pointermove", (e) => {
      // Track the pointer for hover raycasts even when not dragging.
      const r = el.getBoundingClientRect();
      this._mouse.px = e.clientX - r.left;
      this._mouse.py = e.clientY - r.top;
      this._mouse.x = (this._mouse.px / r.width) * 2 - 1;
      this._mouse.y = -(this._mouse.py / r.height) * 2 + 1;
      this._mouse.over = true;
      if (!this._dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      this._orbitGoal.yaw -= dx * 0.005;
      this._orbitGoal.el = clamp(this._orbitGoal.el + dy * 0.004, 0.18, 1.38);
    });
    el.addEventListener("pointerleave", () => { this._mouse.over = false; });
    const stop = (e) => {
      this._dragging = false;
      el.style.cursor = "grab";
      if (e.pointerId != null && el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
    };
    el.addEventListener("pointerup", stop);
    el.addEventListener("pointercancel", stop);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this._orbitGoal.dist = clamp(
        this._orbitGoal.dist * Math.exp(e.deltaY * 0.0011), 26, 150);
    }, { passive: false });
    el.addEventListener("dblclick", () => {
      Object.assign(this._orbitGoal, this._defaultView);
    });
  }

  _buildTooltip() {
    this.tooltip = document.createElement("div");
    this.tooltip.className = "sim3d-tooltip";
    this.tooltip.style.cssText =
      "position:absolute;display:none;pointer-events:none;z-index:60;" +
      "background:rgba(8,15,12,.92);border:1px solid #1e3a2a;border-radius:9px;" +
      "padding:8px 10px;color:#e6f5ec;font:11px ui-monospace,monospace;" +
      "line-height:1.5;white-space:nowrap;box-shadow:0 10px 30px #000a;" +
      "backdrop-filter:blur(6px);";
    this.root.appendChild(this.tooltip);
  }

  // Hover-for-stats: everything shown is read straight from authoritative
  // state (position, velocity, flags) or the plan steps — nothing invented.
  _updateHover(sim) {
    if (!this._mouse.over || this._dragging) return this._setHover(null);
    this.raycaster.setFromCamera(this._mouse, this.camera);
    const bodies = [];
    for (const t of this.tokens.values()) bodies.push(t.body);
    const hit = this.raycaster.intersectObjects(bodies, false)[0];
    if (!hit) return this._setHover(null);
    const p = (this.state.players || []).find(
      (q) => q.id === hit.object.userData.playerId);
    if (!p) return this._setHover(null);

    this._setHover(p.id);
    const isAttack = p.team === sim.attackingTeam;
    const color = p.team === "home" ? HOME : AWAY;
    const speed = Math.hypot(p.vx || 0, p.vy || 0);
    const b = this.state.ball || { x: p.x, y: p.y };
    const toBall = Math.hypot(p.x - b.x, p.y - b.y);

    const lines = [
      `<b style="color:${color}">#${p.number}</b> · ` +
      `${isAttack ? "attacking" : "defending"}`,
      `${speed.toFixed(1)} m/s · ${toBall.toFixed(0)} m from ball`,
    ];
    if (isAttack && p.number === sim.ballOwnerNumber) {
      lines.push(`<span style="color:${BALL}">● on the ball</span>`);
    }
    if (isAttack) {
      for (const s of sim.steps || []) {
        const pct = s.successProbability != null
          ? ` · ${Math.round(s.successProbability * 100)}%` : "";
        if (s.fromNumber === p.number) {
          lines.push(`${s.type === "shot" ? "shoots" : "plays"} step ${s.index + 1}${pct}`);
        } else if (s.toNumber === p.number) {
          lines.push(`receives step ${s.index + 1}${pct}`);
        }
      }
    }
    if (p.edited) lines.push(`<span style="color:#a7f3d0">repositioned by coach</span>`);

    this.tooltip.innerHTML = lines.join("<br>");
    this.tooltip.style.display = "block";
    const w = this.tooltip.offsetWidth, h = this.tooltip.offsetHeight;
    const maxX = (this.mount.clientWidth || 0) - w - 8;
    const x = Math.min(this._mouse.px + 16, Math.max(8, maxX));
    const y = Math.max(8, this._mouse.py - h - 12);
    this.tooltip.style.left = `${x}px`;
    this.tooltip.style.top = `${y}px`;
  }

  _setHover(id) {
    if (this._hoverId === id) return;
    const prev = this.tokens.get(this._hoverId);
    if (prev) prev.key = "";           // force material refresh next frame
    this._hoverId = id;
    if (id == null) this.tooltip.style.display = "none";
    if (!this._dragging) this.root.style.cursor = id != null ? "pointer" : "grab";
  }

  _updateCamera() {
    const o = this.orbit, g = this._orbitGoal;
    o.yaw += (g.yaw - o.yaw) * 0.12;
    o.el += (g.el - o.el) * 0.12;
    o.dist += (g.dist - o.dist) * 0.12;
    const cy = Math.cos(o.el) * o.dist;
    this.camera.position.set(
      Math.sin(o.yaw) * cy,
      Math.sin(o.el) * o.dist,
      Math.cos(o.yaw) * cy);
    this.camera.lookAt(0, 0, 0);
  }

  // -- lifecycle ------------------------------------------------------------
  show() {
    if (this.visible) return;
    this.visible = true;
    this.root.style.display = "block";
    // Intro: drop in from a high top-down view onto the touchline angle.
    Object.assign(this.orbit, { yaw: 0.0, el: 1.35, dist: 130 });
    Object.assign(this._orbitGoal, this._defaultView);
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.root.style.display = "none";
    this._setHover(null);
  }

  applyState(state) {
    this.state = state;
    if (state?.pitch) { this.pitchL = state.pitch.length; this.pitchW = state.pitch.width; }
    const active = !!state?.simulation?.active;
    if (active) this.show(); else this.hide();
  }

  // -- per-frame ------------------------------------------------------------
  _resize() {
    const w = this.mount.clientWidth || 960;
    const h = this.mount.clientHeight || 600;
    if (w === this._lastW && h === this._lastH) return;
    this._lastW = w; this._lastH = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _wx(x) { return x - this.pitchL / 2; }
  _wz(y) { return y - this.pitchW / 2; }

  _frame() {
    this._raf = requestAnimationFrame(() => this._frame());
    if (!this.visible || !this.state) return;
    this._resize();

    const sim = this.state.simulation || {};
    const attack = sim.attackingTeam;
    const carrier = sim.ballOwnerNumber;
    const seen = new Set();
    let carrierToken = null;

    for (const p of this.state.players || []) {
      seen.add(p.id);
      const t = this._token(p);
      t.cur.x += (p.x - t.cur.x) * LERP;
      t.cur.y += (p.y - t.cur.y) * LERP;
      t.group.position.set(this._wx(t.cur.x), 0, this._wz(t.cur.y));

      const isAttack = p.team === attack;
      const isCarrier = isAttack && p.number === carrier;
      const isHover = p.id === this._hoverId;
      if (isCarrier) carrierToken = t;
      const key = `${p.team}:${isAttack}:${isCarrier}:${isHover}:${p.number}`;
      if (t.key !== key) {
        t.key = key;
        const base = p.team === "home" ? HOME : AWAY;
        const mat = t.body.material;
        mat.color.set(base);
        mat.transparent = !isAttack;
        mat.opacity = isAttack ? 1 : 0.82;
        mat.emissive.set(isCarrier ? 0x6b5e00 : isHover ? 0x274d3d : 0x000000);
      }
    }
    for (const [id, t] of this.tokens) {
      if (!seen.has(id)) {
        this.scene.remove(t.group);
        this.tokens.delete(id);
      }
    }

    const b = this.state.ball || { x: 52.5, y: 34 };
    this.ballCur.x += (b.x - this.ballCur.x) * (LERP + 0.1);
    this.ballCur.y += (b.y - this.ballCur.y) * (LERP + 0.1);
    this.ballMesh.position.set(this._wx(this.ballCur.x), 0.34, this._wz(this.ballCur.y));

    if (carrierToken) {
      this.ring.visible = true;
      this.ring.position.set(
        carrierToken.group.position.x, 0.05, carrierToken.group.position.z);
    } else {
      this.ring.visible = false;
    }

    this._drawLane(sim);
    this._updateHover(sim);
    this._updateCamera();
    this.renderer.render(this.scene, this.camera);
  }

  // The active step's intended pass, drawn on the grass between the two
  // players' *live* positions — same reading of state as the CSS version.
  _drawLane(sim) {
    const step = (sim.steps || []).find((s) => s.status === "active");
    const mat = this.lane.material;
    if (!step || step.type !== "pass" || step.toNumber == null) {
      mat.opacity = 0;
      return;
    }
    const players = this.state.players || [];
    const from = players.find((p) => p.team === sim.attackingTeam && p.number === step.fromNumber);
    const to = players.find((p) => p.team === sim.attackingTeam && p.number === step.toNumber);
    if (!from || !to) { mat.opacity = 0; return; }
    const x1 = this._wx(from.x), z1 = this._wz(from.y);
    const x2 = this._wx(to.x), z2 = this._wz(to.y);
    const len = Math.hypot(x2 - x1, z2 - z1);
    if (len < 0.5) { mat.opacity = 0; return; }
    mat.color.set(sim.attackingTeam === "home" ? HOME : AWAY);
    mat.opacity = 0.8;
    this.lane.scale.set(len, 1.1, 1);
    this.lane.position.set((x1 + x2) / 2, 0.04, (z1 + z2) / 2);
    // PlaneGeometry lies in XY before the -90° X rotation, so its local +x
    // maps to world +x and local +y to world -z: yaw is atan2 of (-dz, dx).
    this.lane.rotation.z = Math.atan2(-(z2 - z1), x2 - x1);
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

/* ------------------------------------------------------------------------- *
 * CSS-3D fallback — the original implementation, unchanged behaviour.
 * ------------------------------------------------------------------------- */
const THETA = 56;            // pitch tilt in degrees

export class CssSim3DView {
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
