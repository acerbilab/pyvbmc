// Record an animation page through its capture hook, as an MP4, a GIF or a folder of PNG frames.
//
//   node scripts/record.mjs PAGE OUT [--from S] [--to S] [--fps N] [--size WxH] [--controls]
//                                    [--crf N] [--denoise L:C:LT:CT] [--gif-width N] [--colors N] [--bayer N]
//
// PAGE is a page of the folder above this one, with any query parameters ("wordmark.html",
// "index.html?hud=0"); the script serves that folder itself and adds capture=1. OUT ending in .mp4 or
// .gif is encoded by ffmpeg (FFMPEG, or ffmpeg on the PATH); any other OUT is a folder that receives
// f0000.png, f0001.png, ... Chrome is CHROME, or its default install location. Needs Node 22 or later
// (global WebSocket) and nothing else.
//
// The page draws a frame on request, vbmcCapture.frame(t, dt). The recording steps it from --from
// (default 0, the title card) to --to (default the end of the loop; a later time wraps into the next
// loop) with dt = 1 / fps. Only a recording that starts at 0 is the page as it plays: one that starts
// later begins with the camera where that segment wants it, not where playback would have left it.
// The playback controls are hidden unless --controls is given; the page's hud=0 hides the captions too.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const USAGE = "usage: node scripts/record.mjs PAGE OUT [--from S] [--to S] [--fps N] [--size WxH] [--controls] "
  + "[--crf N] [--denoise L:C:LT:CT] [--gif-width N] [--colors N] [--bayer N]";
const pos = [], opt = {};
for (const argv = process.argv.slice(2); argv.length;) {
  const a = argv.shift();
  if (a === "--controls") opt.controls = true;
  else if (a.startsWith("--")) opt[a.slice(2)] = argv.shift();
  else pos.push(a);
}
if (pos.length !== 2) { console.error(USAGE); process.exit(2); }
const [page, out] = pos;
const kind = /\.mp4$/i.test(out) ? "mp4" : /\.gif$/i.test(out) ? "gif" : "frames";
const FPS = Number(opt.fps || (kind === "gif" ? 15 : 30));
const [W, H] = (opt.size || "1280x720").split("x").map(Number);
const CHROME = process.env.CHROME || {
  win32: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  darwin: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
}[process.platform] || "google-chrome";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The folder, served over HTTP: the page loads its trace with a script tag, which file:// may refuse.
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8" };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  try {
    if (!path.startsWith(ROOT + sep) || !statSync(path).isFile()) throw new Error();
    res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" }).end(readFileSync(path));
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/${page}${page.includes("?") ? "&" : "?"}capture=1`;

// Headless Chrome with a fresh profile (a cached trace would outlive a regeneration), over the DevTools protocol.
const profile = mkdtempSync(join(tmpdir(), "vbmc3d-record-"));
const port = 9300 + Math.floor(Math.random() * 600);
const chrome = spawn(CHROME, ["--headless=new", "--enable-unsafe-swiftshader", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
let target;
for (let k = 0; k < 100 && !target; k++) {
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); }
  catch { await sleep(200); }
}
if (!target) throw new Error(`Chrome (${CHROME}) did not open its DevTools port`);
const sock = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r, j) => { sock.addEventListener("open", r); sock.addEventListener("error", j); });
let nextId = 0; const pending = new Map();
sock.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; pending.set(id, r); sock.send(JSON.stringify({ id, method, params })); });
async function evaluate(expression) {
  const m = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (m.result?.exceptionDetails) throw new Error(`page: ${m.result.exceptionDetails.exception?.description || expression}`);
  return m.result?.result?.value;
}

await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
let ready = false;
for (let k = 0; k < 600 && !ready; k++) { ready = await evaluate("!!window.vbmcCapture").catch(() => false); if (!ready) await sleep(100); }
if (!ready) throw new Error(`${url} did not start (is it one of the animation pages?)`);
await evaluate("document.fonts.ready.then(() => true)");   // the captions' typefaces; the letters' has been waited for
if (!opt.controls) await evaluate(`document.head.appendChild(Object.assign(document.createElement("style"),
  { textContent: "#transport, #chips { display: none !important; }" })) && true`);
const END = await evaluate("vbmcCapture.end");
const T0 = Number(opt.from || 0), T1 = opt.to !== undefined ? Number(opt.to) : END;
const n = Math.max(1, Math.round((T1 - T0) * FPS));

let sink;
if (kind === "frames") {
  mkdirSync(out, { recursive: true });
  let k = 0;
  sink = { write: async (png) => writeFileSync(join(out, `f${String(k++).padStart(4, "0")}.png`), png), close: async () => {} };
} else {
  const denoise = opt.denoise || (kind === "gif" ? "6:5:10:10" : "");
  const filter = kind === "gif"
    ? `scale=${opt["gif-width"] || 640}:-1:flags=lanczos,hqdn3d=${denoise},split[a][b];`
      + `[a]palettegen=max_colors=${opt.colors || 96}:stats_mode=diff[p];`
      + `[b][p]paletteuse=dither=bayer:bayer_scale=${opt.bayer || 3}:diff_mode=rectangle`
    : denoise ? `hqdn3d=${denoise}` : "";
  const encode = kind === "gif" ? ["-loop", "0"]
    : ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", String(opt.crf || 18), "-preset", "slow", "-movflags", "+faststart"];
  const ff = spawn(process.env.FFMPEG || "ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS),
    "-c:v", "png", "-i", "-", ...(filter ? ["-vf", filter] : []), ...encode, out], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((r, j) => { ff.on("close", (code) => (code ? j(new Error(`ffmpeg exited with ${code}`)) : r())); ff.on("error", j); });
  sink = {
    write: (png) => new Promise((r) => (ff.stdin.write(png) ? r() : ff.stdin.once("drain", r))),
    close: () => { ff.stdin.end(); return done; },
  };
}

const started = Date.now();
for (let k = 0; k < n; k++) {
  await evaluate(k ? `vbmcCapture.frame(${T0 + k / FPS}, ${1 / FPS})` : `vbmcCapture.frame(${T0}, 0)`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  await sink.write(Buffer.from(shot.result.data, "base64"));
  if (k % 150 === 0) console.log(`frame ${k} of ${n}, t = ${(T0 + k / FPS).toFixed(2)} s of a ${END.toFixed(2)} s loop (${((Date.now() - started) / 1000).toFixed(0)} s)`);
}
await sink.close();
console.log(`wrote ${out}: ${n} frames at ${FPS} fps, ${W} x ${H}, in ${((Date.now() - started) / 1000).toFixed(0)} s`);
sock.close(); chrome.kill(); server.close();
await sleep(500);
try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold it on Windows */ }
