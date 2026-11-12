// Run the golden fixtures against widget functions. Usage: node widget-harness.js <file.js | page.html>
// Compares only the keys listed under "out" (message text is free).
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const args = process.argv.slice(2);
const ri = args.indexOf("--require");
const required = ri >= 0 ? args.splice(ri, 2)[1].split(",") : [];
const file = args[0];
if (!file) { console.error("usage: node widget-harness.js <file.js|page.html>"); process.exit(2); }
let code = fs.readFileSync(file, "utf8");
if (file.endsWith(".html")) {
  const m = [...code.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
  if (!m.length) { console.error("no inline script in " + file); process.exit(1); }
  code = m.map((x) => x[1]).join("\n");
}
const ctx = vm.createContext({});
vm.runInContext(code + "\nthis.__fns = {checkResult: typeof checkResult==='function'?checkResult:null, strategyRows: typeof strategyRows==='function'?strategyRows:null};", ctx);
const fns = ctx.__fns;
const golden = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "widgets.golden.json"), "utf8"));
let bad = 0, run = 0;

function same(want, got) { return JSON.stringify(JSON.parse(JSON.stringify(got))) === JSON.stringify(want); }

for (const name of ["checkResult", "strategyRows"]) {
  if (!fns[name]) {
    if (required.includes(name)) { bad++; console.log(`FAIL ${name} is required but not defined at top level`); }
    else console.log(`skip ${name} (not defined in this file)`);
    continue;
  }
  for (const c of golden[name]) {
    run++;
    const args = name === "checkResult" ? [c.in] : c.in;
    let got, threw = false;
    try { got = fns[name](...args); } catch (e) { threw = true; }
    let ok;
    if (c.throws) ok = threw;
    else if (threw) ok = false;
    else if (name === "checkResult") ok = got.status === c.out.status;
    else ok = same(c.out, got);
    if (!ok) { bad++; console.log(`FAIL ${name}(${JSON.stringify(c.in)})`); }
  }
}
console.log(bad ? `${bad} of ${run} failed` : `ok (${run} cases)`);
process.exit(bad ? 1 : 0);
