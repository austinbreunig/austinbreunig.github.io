// Stub for the plumber widget functions. Returns the golden answers for known inputs only.
// Page tickets (#33 #34 #35) replace this with real functions inline in projects/plumber.html.
var PHASES = { load: { signature: "run(gdf, **params) -> gdf", config: "- name: load" } };

function phaseInfo(name) {
  if (!PHASES[name]) throw new Error("unknown phase: " + name);
  return PHASES[name];
}

function checkResult(choice) {
  var out = {
    "good": { status: "PASS", message: "stub" },
    "wrong-signature": { status: "FAIL", message: "stub" },
    "bad-crs": { status: "WARN", message: "stub" },
  };
  if (!out[choice]) throw new Error("unknown choice: " + choice);
  return out[choice];
}

function strategyRows(strategy, rows, partitions) {
  if (strategy !== "local" && strategy !== "localmp") throw new Error("unknown strategy");
  if (!(partitions >= 1)) throw new Error("partitions must be >= 1");
  var parts = [];
  for (var i = 0; i < (strategy === "local" ? 1 : partitions); i++) parts.push([]);
  rows.forEach(function (r, i) { parts[i % parts.length].push(r); });
  return { yamlLine: "strategy: " + strategy, partitions: parts, returned: rows.slice() };
}
