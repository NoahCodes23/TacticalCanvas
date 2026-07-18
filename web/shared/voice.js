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
  }

  get active() {
    return !!this.conversation;
  }

  async start() {
    if (this.conversation || this.starting) return;
    this.starting = true;
    this.onStatus("connecting");
    try {
      // Ask before connecting. Prompting first means a denied mic surfaces as a
      // clear message instead of a session that connects and then hears nothing.
      await navigator.mediaDevices.getUserMedia({ audio: true });

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

      this.conversation = await ElevenLabsClient.Conversation.startSession({
        signedUrl: data.signedUrl,
        connectionType: "websocket",
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

  async toggle() {
    if (this.conversation || this.starting) await this.stop();
    else await this.start();
  }
}
