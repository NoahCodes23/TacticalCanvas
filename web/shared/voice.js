/**
 * Browser voice session for the dashboard's microphone button.
 *
 * The split is the same one voice_agent.py uses, just moved into the page:
 * ElevenLabs handles speech-to-text, text-to-speech and turn-taking; the agent
 * itself is configured to think with OpenRouter (tencent/hy3), so no ElevenLabs
 * LLM is involved; and the tool calls land back here as plain JS functions.
 *
 * Tools are supplied by the caller because only the dashboard has the pieces
 * they need -- the live state snapshot and the open command WebSocket. That
 * keeps this module free of board knowledge.
 *
 * The API key never reaches the browser: /api/voice-token mints a short-lived
 * signed URL server-side and we connect with that.
 */

export class VoiceSession {
  /**
   * @param {object} options
   * @param {Record<string, (params: object) => Promise<string>|string>} options.clientTools
   *   Tool handlers keyed by the tool name registered on the ElevenLabs agent.
   * @param {(status: string) => void} [options.onStatus]  "connecting" | "connected" | "disconnected" | "error"
   * @param {(mode: string) => void} [options.onMode]      "speaking" | "listening"
   * @param {(role: string, text: string) => void} [options.onTranscript]
   * @param {(message: string) => void} [options.onError]
   */
  constructor({ clientTools, onStatus, onMode, onTranscript, onError }) {
    this.clientTools = clientTools || {};
    this.onStatus = onStatus || (() => {});
    this.onMode = onMode || (() => {});
    this.onTranscript = onTranscript || (() => {});
    this.onError = onError || (() => {});
    this.conversation = null;
    this.starting = false;
    /** "webrtc", "websocket", or "text", once connected. */
    this.transport = null;
    /** True when the live session never opened the mic. */
    this.textOnly = false;
    /** True while the live session's mic is muted (push-to-talk idle). */
    this.micMuted = false;
  }

  get active() {
    return !!this.conversation;
  }

  /**
   * @param {object} [options]
   * @param {boolean} [options.textOnly]  Connect without ever enabling the mic.
   *   The typed fallback uses this: the SDK skips setMicrophoneEnabled entirely,
   *   so a noisy room cannot reach the agent at all -- no filtering to lose an
   *   argument with. The agent still runs tools and still answers, in text.
   * @param {boolean} [options.startMuted]  Connect with the mic muted, for
   *   push-to-talk. Unlike textOnly this keeps a real voice session, so the mic
   *   can be opened for a turn and closed again without reconnecting.
   */
  async start({ textOnly = false, startMuted = false } = {}) {
    if (this.conversation || this.starting) return;
    this.starting = true;
    this.textOnly = textOnly;
    this.onStatus("connecting");
    try {
      // Ask before connecting. Prompting first means a denied mic surfaces as a
      // clear message instead of a session that connects and then hears nothing.
      // Release it immediately: the SDK opens its own stream with its own noise
      // filtering, and a second live track held here would keep the mic hot for
      // the life of the page. Skipped for text-only: asking for a microphone we
      // will never open is how you get a permission prompt nobody expected.
      if (!textOnly) {
        const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
        for (const track of probe.getTracks()) track.stop();
      }

      const response = await fetch("/api/voice-token");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Could not start a voice session.");

      // Wrap every handler: a tool that throws must be reported back to the
      // model as text it can speak, never left as a rejected promise that
      // stalls the turn.
      const wrapped = {};
      for (const [name, handler] of Object.entries(this.clientTools)) {
        wrapped[name] = async (params) => {
          try {
            const result = await handler(params || {});
            return typeof result === "string" ? result : JSON.stringify(result);
          } catch (error) {
            return `Tool ${name} failed: ${error.message || error}`;
          }
        };
      }

      // Transport choice, and what it does and does not buy.
      //
      // Both paths get the same microphone processing, and it is the *browser's*,
      // not ElevenLabs': the SDK opens getUserMedia with voiceIsolation,
      // echoCancellation, noiseSuppression and autoGainControl hardcoded, on
      // every path, with no way to override them. There is no ElevenLabs noise
      // model in the realtime input path -- their voice isolator is a separate
      // file-based API, and the Krisp hooks in the bundle are LiveKit reporting
      // plumbing with no processor attached.
      //
      // WebRTC is still the better transport (jitter buffering, packet loss
      // recovery, tighter turn-taking), so it stays the default -- but it is not
      // a background-noise fix. The typed path below is the actual answer to a
      // loud room.
      //
      // The two credentials are not interchangeable: the SDK throws if a signed
      // URL is paired with connectionType "webrtc". Text-only forces the
      // WebSocket path, since there is no mic to carry.
      const attempts = [];
      if (data.conversationToken && !textOnly) {
        attempts.push({
          label: "webrtc",
          config: { conversationToken: data.conversationToken, connectionType: "webrtc" },
        });
      }
      if (data.signedUrl) {
        attempts.push({
          label: textOnly ? "text" : "websocket",
          config: { signedUrl: data.signedUrl, connectionType: "websocket", textOnly },
        });
      }
      if (!attempts.length) throw new Error("Could not start a voice session.");

      let lastError = null;
      for (const attempt of attempts) {
        try {
          this.conversation = await this._startSession(attempt.config, wrapped);
          this.transport = attempt.label;
          break;
        } catch (error) {
          lastError = error;
          this.conversation = null;
        }
      }
      if (!this.conversation) throw lastError || new Error("Could not start a voice session.");
      // Mute before the coach has said anything, so a session opened in a noisy
      // room never gets a chance to transcribe the room.
      if (startMuted && !textOnly) this.setMicMuted(true);
    } catch (error) {
      this.conversation = null;
      const message =
        error?.name === "NotAllowedError"
          ? "Microphone permission denied."
          : error?.message || String(error);
      this.onError(message);
      this.onStatus("error");
    } finally {
      this.starting = false;
    }
  }

  /** One startSession attempt for a given transport config. */
  async _startSession(transportConfig, wrapped) {
    return ElevenLabsClient.Conversation.startSession({
      ...transportConfig,
      clientTools: wrapped,
      onConnect: () => this.onStatus("connected"),
      onDisconnect: () => {
        this.conversation = null;
        this.onStatus("disconnected");
      },
      onModeChange: ({ mode }) => this.onMode(mode),
      onError: (message) => this.onError(String(message)),
      onMessage: ({ message, source }) => {
        if (message) this.onTranscript(source === "user" ? "user" : "agent", message);
      },
    });
  }

  async stop() {
    const session = this.conversation;
    this.conversation = null;
    if (session) {
      try {
        await session.endSession();
      } catch {
        // Already gone; the status callback has fired either way.
      }
    }
    this.onStatus("disconnected");
  }

  async toggle(options = {}) {
    // A live text-only session has no mic, so the mic button must upgrade it
    // rather than read as "already on" and switch everything off.
    if (this.conversation && this.textOnly) {
      await this.stop();
      await this.start(options);
      return;
    }
    if (this.conversation || this.starting) await this.stop();
    else await this.start(options);
  }

  /**
   * Open or close the microphone on a live voice session.
   *
   * This is what makes push-to-talk possible without reconnecting: the session,
   * its context and its tools all stay up while the mic is closed, so the room
   * is simply not being listened to between turns. Text-only sessions have no
   * mic to mute and the SDK throws on them, hence the guard.
   *
   * @returns {boolean} whether the mute state was applied.
   */
  setMicMuted(muted) {
    if (!this.conversation || this.textOnly) return false;
    try {
      this.conversation.setMicMuted(muted);
      this.micMuted = muted;
      return true;
    } catch (error) {
      this.onError(error?.message || String(error));
      return false;
    }
  }

  /**
   * Send a typed turn to the agent, exactly as if it had been spoken.
   *
   * Opens a text-only session if nothing is connected, so typing works without
   * ever touching the microphone -- the point of the whole path. If a voice
   * session is already live it reuses it, so you can talk and fall back to
   * typing mid-conversation without losing context.
   *
   * @returns {Promise<boolean>} whether the message went out.
   */
  async sendText(text) {
    const message = String(text ?? "").trim();
    if (!message) return false;

    if (!this.conversation) {
      await this.start({ textOnly: true });
      if (!this.conversation) return false;  // start() already surfaced why
    }
    try {
      this.conversation.sendUserMessage(message);
      this.onTranscript("user", message);
      return true;
    } catch (error) {
      this.onError(error?.message || String(error));
      return false;
    }
  }
}
