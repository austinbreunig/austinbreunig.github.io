// Run: node docs/contracts/embed.test.js
// Runs embed.js against a tiny fake DOM and checks the S2 protocol.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function el(attrs, sets) {
  const a = Object.assign({}, attrs);
  return {
    hidden: false, style: {}, listeners: {},
    getAttribute: (k) => a[k], setAttribute: (k, v) => { a[k] = v; sets.push(v); },
    addEventListener(t, f) { this.listeners[t] = f; },
  };
}

function run() {
  const sets = [];
  const frame = el({}, sets), iframe = el({ src: "../visuals/x.html" }, sets), fallback = el({}, sets), reload = el({}, sets);
  iframe.contentWindow = {};
  frame.querySelector = () => iframe;
  const figure = { querySelector: (s) => ({ ".visual-frame": frame, ".visual-fallback": fallback, ".visual-reload": reload })[s] };
  let onMessage; const timers = [];
  const ctx = {
    document: { querySelectorAll: () => [figure] },
    window: { addEventListener: (t, f) => (onMessage = f) },
    setTimeout: (f, ms) => { timers.push({ f, ms, live: true }); return timers.length - 1; },
    clearTimeout: (i) => { if (timers[i]) timers[i].live = false; },
    Number,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../../embed.js"), "utf8"), ctx);
  return { frame, iframe, fallback, reload, onMessage, timers, sets };
}

// 4 s timeout shows fallback
let t = run();
assert.strictEqual(t.timers[0].ms, 4000);
t.timers[0].f();
assert(t.frame.hidden && !t.fallback.hidden);

// Reload restores the frame, resets src, restarts the timer
t.reload.listeners.click();
assert(!t.frame.hidden && t.fallback.hidden);
assert.deepStrictEqual(t.sets, ["../visuals/x.html"]);
assert.strictEqual(t.timers.length, 2);

// ready cancels the timer; height sets px; foreign source and bad px ignored
t = run();
t.onMessage({ source: {}, data: { type: "visual-ready" } });
assert(t.timers[0].live);
t.onMessage({ source: t.iframe.contentWindow, data: { type: "visual-height", px: 321 } });
assert.strictEqual(t.iframe.style.height, "321px");
t.onMessage({ source: t.iframe.contentWindow, data: { type: "visual-height", px: "9" } });
assert.strictEqual(t.iframe.style.height, "321px");
t.onMessage({ source: t.iframe.contentWindow, data: { type: "visual-ready" } });
assert(!t.timers[0].live);
console.log("embed tests ok");
