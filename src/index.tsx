import { createSignal, createResource, createEffect } from "solid-js";
import { render } from "solid-js/web";
import { runBenchmark } from "./benchmark";
import "./index.css";

// WebSocket console forwarding for external monitoring
if (new URLSearchParams(window.location.search).has("benchmark")) {
  const ws = new WebSocket("ws://localhost:8765");
  const originalLog = console.log;
  const originalWarn = console.warn;
  const browser = /Firefox/.test(navigator.userAgent)
    ? "Firefox"
    : /Chrome/.test(navigator.userAgent)
      ? "Chrome"
      : /Safari/.test(navigator.userAgent)
        ? "Safari"
        : navigator.userAgent;
  ws.onopen = () => console.log(`[${browser}] Connected`);
  ws.onerror = () => {};
  console.log = (...args) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(`[${browser}] ` + args.map((a) => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" "));
    }
    originalLog.apply(console, args);
  };
  console.warn = (...args) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(`[${browser}] [WARN] ` + args.map((a) => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" "));
    }
    originalWarn.apply(console, args);
  };
}

const App = () => {
  // Original used DOWNSAMPLE=2: 1124>>2=281, 854>>2=213
  const [width] = createSignal(281);
  const [height] = createSignal(213);

  let canvas!: HTMLCanvasElement;

  const [gpu] = createResource(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter");
    const canTimestamp = adapter.features.has("timestamp-query");
    if (!canTimestamp) throw new Error("timestamp-query not supported");
    return adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  });

  createEffect(() => {
    const device = gpu();
    if (!device) return;

    const context = canvas.getContext("webgpu")!;
    context.configure({
      device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });

    const doBenchmark = () => runBenchmark(device, width(), height());

    window.addEventListener("keydown", (e) => {
      if (e.key === "b") doBenchmark();
    });

    // Auto-run benchmark if ?benchmark is in URL
    if (new URLSearchParams(window.location.search).has("benchmark")) {
      setTimeout(doBenchmark, 500);
    }
  });

  return <canvas ref={canvas} width={width()} height={height()}></canvas>;
};

render(App, document.getElementById("root")!);
