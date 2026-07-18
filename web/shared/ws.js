const PROTOCOL_VERSION = 1;

export class Connection {
  constructor(clientId) {
    this.clientId = clientId;
    this.seq = 0;
    this.ws = null;
    this.rtt = 0;
    this.serverToBrowserMs = 0;
    this.captureToBrowserMs = 0;
    this.connected = false;
    this.reconnects = 0;
    this.state = null;
    this.onState = () => {};
    this.onError = () => {};
    this._backoff = 250;
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.ws = new WebSocket(`${proto}//${location.host}/ws`);

    this.ws.onopen = () => {
      this.connected = true;
      this._backoff = 250;
      this._pingTimer = setInterval(() => this.ping(), 1000);
    };

    this.ws.onmessage = (e) => {
      const receivedAt = Date.now();
      const msg = JSON.parse(e.data);
      if (msg.type === "STATE_SNAPSHOT" || msg.type === "VISION_UPDATE") {
        this.serverToBrowserMs = Math.max(0, receivedAt - msg.timestamp);
        const newestCapture = Math.max(
          msg.payload.vision?.capturedAtMs || 0,
          0,
          ...(msg.payload.cursors || []).map((cursor) => cursor.capturedAtMs || 0),
        );
        if (newestCapture) {
          this.captureToBrowserMs = Math.max(0, receivedAt - newestCapture);
        }
        this.state = msg.type === "STATE_SNAPSHOT"
          ? msg.payload
          : { ...(this.state || {}), ...msg.payload };
        this.onState(this.state);
      }
      else if (msg.type === "PONG") this.rtt = performance.now() - msg.payload.t;
      else if (msg.type === "ERROR") this.onError(msg.payload.reason);
    };

    this.ws.onclose = () => {
      this.connected = false;
      clearInterval(this._pingTimer);
      setTimeout(() => {
        this.reconnects++;
        this.connect();
      }, this._backoff);
      this._backoff = Math.min(this._backoff * 2, 4000);
    };
  }

  send(type, payload = {}) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      scenarioId: "demo",
      clientId: this.clientId,
      sequenceNumber: ++this.seq,
      timestamp: Date.now(),
      type,
      payload,
    }));
  }

  ping() {
    this.send("PING", { t: performance.now() });
  }
}
