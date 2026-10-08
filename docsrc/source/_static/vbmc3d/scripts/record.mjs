// Record an animation page through its capture hook, as an MP4, a GIF or a folder of PNG frames.
// With OUT ending in .json, write instead the JSON that the page puts in its <pre id="events">
// (film.html?events=1: the times that scripts/make_score.py follows).
//
//   node scripts/record.mjs PAGE OUT [--from S] [--replay] [--to S] [--fps N] [--size WxH] [--scale K] [--controls]
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
// --replay (film.html, whose vbmcCapture.step advances its state without drawing) first steps the page
// through every frame before --from, rounded to a frame, as a recording from 0 would have drawn them:
// the stretch it then records is the page as it plays, and can be spliced into a recording from 0.
// The playback controls are hidden unless --controls is given; the page's hud=0 hides the captions too.
//
// --size is the page's layout in CSS pixels, and --scale (default 1) its device pixel ratio, so that the
// frames are size x scale pixels. The text and readouts are sized in CSS pixels, so a larger --size
// shrinks them against the frame, while --scale keeps the layout of --size and draws it sharper:
// --size 1280x720 --scale 1.5 gives frames of 1920 x 1080 that are composed as at 1280 x 720.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const USAGE = "usage: node scripts/record.mjs PAGE OUT [--from S] [--replay] [--to S] [--fps N] [--size WxH] [--scale K] "
  + "[--controls] [--crf N] [--denoise L:C:LT:CT] [--gif-width N] [--colors N] [--bayer N]";
const pos = [], opt = {};
for (const argv = process.argv.slice(2); argv.length;) {
  const a = argv.shift();
  if (a === "--controls" || a === "--replay") opt[a.slice(2)] = true;
  else if (a.startsWith("--")) opt[a.slice(2)] = argv.shift();
  else pos.push(a);
}
if (pos.length !== 2) { console.error(USAGE); process.exit(2); }
const [page, out] = pos;
const kind = /\.mp4$/i.test(out) ? "mp4" : /\.gif$/i.test(out) ? "gif" : /\.json$/i.test(out) ? "json" : "frames";
const FPS = Number(opt.fps || (kind === "gif" ? 15 : 30));
const [W, H] = (opt.size || "1280x720").split("x").map(Number);
const SCALE = Number(opt.scale || 1);
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

await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: SCALE, mobile: false });
await send("Page.navigate", { url });
let ready = false;
for (let k = 0; k < 600 && !ready; k++) { ready = await evaluate("!!window.vbmcCapture").catch(() => false); if (!ready) await sleep(100); }
if (!ready) throw new Error(`${url} did not start (is it one of the animation pages?)`);
await evaluate("document.fonts.ready.then(() => true)");   // the captions' typefaces; the letters' has been waited for
if (!opt.controls) await evaluate(`document.head.appendChild(Object.assign(document.createElement("style"),
  { textContent: "#transport, #chips { display: none !important; }" })) && true`);
if (kind === "json") {
  const text = await evaluate(`document.getElementById("events")?.textContent ?? null`);
  if (!text) throw new Error(`${page} wrote no events (film.html?events=1 does)`);
  writeFileSync(out, text); console.log(`wrote ${out}`);
  sock.close(); chrome.kill(); server.close(); await sleep(500);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold it on Windows */ }
  process.exit(0);
}
const END = await evaluate("vbmcCapture.end");
// With --replay, frame k of the recording is frame K0 + k of a recording from 0, at the same time and dt.
const K0 = opt.replay ? Math.round(Number(opt.from || 0) * FPS) : 0;
const T0 = opt.replay ? K0 / FPS : Number(opt.from || 0), T1 = opt.to !== undefined ? Number(opt.to) : END;
const n = Math.max(1, Math.round((T1 - T0) * FPS));
const at = (k) => (opt.replay ? (K0 + k) / FPS : T0 + k / FPS);
const dtAt = (k) => (K0 + k ? 1 / FPS : 0);
if (opt.replay && K0) {
  if (!(await evaluate("typeof vbmcCapture.step === 'function'"))) throw new Error(`${page} cannot replay (no vbmcCapture.step)`);
  const t = Date.now();
  for (let k = 0; k < K0; k += 300) {
    await evaluate(`for (let k = ${k}; k < ${Math.min(k + 300, K0)}; k++) vbmcCapture.step(k / ${FPS}, k ? 1 / ${FPS} : 0); true`);
  }
  console.log(`replayed frames 0 to ${K0 - 1} in ${((Date.now() - t) / 1000).toFixed(0)} s`);
}

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
  await evaluate(`vbmcCapture.frame(${at(k)}, ${dtAt(k)})`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  await sink.write(Buffer.from(shot.result.data, "base64"));
  if (k % 150 === 0) console.log(`frame ${k} of ${n}, t = ${at(k).toFixed(2)} s of a ${END.toFixed(2)} s loop (${((Date.now() - started) / 1000).toFixed(0)} s)`);
}
await sink.close();
console.log(`wrote ${out}: ${n} frames at ${FPS} fps, ${Math.round(W * SCALE)} x ${Math.round(H * SCALE)}, `
  + `in ${((Date.now() - started) / 1000).toFixed(0)} s`);
sock.close(); chrome.kill(); server.close();
await sleep(500);
try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold it on Windows */ }
