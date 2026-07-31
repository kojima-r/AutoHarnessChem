/**
 * 構造式描画（SmilesDrawer）のブラウザ側の挙動を jsdom で検査し、結果を JSON で出力する。
 *
 *   node tests/js/check_structures.js <index.html> <report.html> <fixtures.json>
 *
 * pytest（tests/test_web_structures.py）から呼ばれる。jsdom が無い環境では skip される。
 * 検査するもの:
 *   - HTML レポート: svg[data-smiles] がすべて描画され（結合線・原子ラベルが生成され）、
 *     反応式（`>` を含む SMILES）も描画されること
 *   - Web UI: SMILES プレビュー / 反応式プレビュー / テンプレート適用 /
 *     プロンプト内 SMILES の検出 / ラン詳細の構造カード（CSV → 反応式）
 */
"use strict";
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const [indexPath, reportPath, fixturesPath] = process.argv.slice(2);
const fixtures = JSON.parse(fs.readFileSync(fixturesPath, "utf-8"));
const libraryPath = path.join(path.dirname(indexPath), "vendor", "smiles-drawer.min.js");
const library = fs.readFileSync(libraryPath, "utf-8");

/** jsdom は canvas と SVGSVGElement.viewBox を持たないため最小限のシムを入れる
 *  （ブラウザではどちらもネイティブに存在する）。 */
function shim(window) {
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "", textAlign: "", textBaseline: "",
    measureText: (t) => ({ width: String(t).length * 7, actualBoundingBoxAscent: 8,
                           actualBoundingBoxDescent: 3 }),
    fillText() {}, clearRect() {}, save() {}, restore() {}, translate() {}, scale() {},
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {},
    arc() {}, setTransform() {}, drawImage() {},
    createLinearGradient: () => ({ addColorStop() {} }),
  });
  const numbers = (value) => String(value || "0 0 0 0").trim().split(/[\s,]+/).map(Number);
  Object.defineProperty(window.SVGSVGElement.prototype, "viewBox", {
    configurable: true,
    get() {
      const [x, y, width, height] = numbers(this.getAttribute("viewBox"));
      return { baseVal: { x, y, width, height } };
    },
  });
  for (const name of ["width", "height"]) {
    Object.defineProperty(window.SVGSVGElement.prototype, name, {
      configurable: true,
      get() { return { baseVal: { value: parseFloat(this.getAttribute(name)) || 0 } }; },
    });
  }
}

function load(file, { url } = {}) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (e) => errors.push(e.message));
  const dom = new JSDOM(fs.readFileSync(file, "utf-8"), {
    runScripts: "outside-only", pretendToBeVisual: true, virtualConsole,
    url: url || "http://127.0.0.1:8000/",
  });
  shim(dom.window);
  return { dom, errors };
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --------------------------------------------------------------- HTML レポート

async function checkReport() {
  const { dom, errors } = load(reportPath);
  const { window } = dom;
  for (const script of window.document.querySelectorAll("script")) {
    window.eval(script.textContent);          // ライブラリ本体は埋め込まれている
  }
  window.document.dispatchEvent(new window.Event("DOMContentLoaded"));
  await wait(600);

  const doc = window.document;
  const targets = [...doc.querySelectorAll("svg[data-smiles]")];
  const reactions = targets.filter((el) => el.getAttribute("data-smiles").includes(">"));
  const tags = {};
  doc.querySelectorAll("svg[data-smiles] *").forEach((el) => {
    tags[el.tagName] = (tags[el.tagName] || 0) + 1;
  });
  const result = {
    library_loaded: typeof window.SmiDrawer === "function",
    total: targets.length,
    drawn: targets.filter((el) => el.childNodes.length > 0).length,
    reactions: reactions.length,
    reactions_drawn: reactions.filter((el) => el.childNodes.length > 0).length,
    bonds: tags.line || 0,
    atom_labels: tags.text || 0,
    render_errors: doc.querySelectorAll(".structure-error").length,
    jsdom_errors: [...new Set(errors)],
  };
  window.close();
  return result;
}

// ------------------------------------------------------------------- Web UI

async function checkUi() {
  const { dom, errors } = load(indexPath);
  const { window } = dom;
  window.eval(library);                       // <script src="/static/..."> の代わり
  window.fetch = async (url) => {
    if (url.includes("artifacts/")) {
      const name = decodeURIComponent(url.split("artifacts/").pop());
      const body = fixtures.csvs[name];
      return { ok: body !== undefined, status: body === undefined ? 404 : 200,
               text: async () => body || "" };
    }
    const routes = {
      "/api/providers": fixtures.providers,
      "/api/runs": { runs: [fixtures.run_summary] },
      [`/api/runs/${fixtures.run_id}`]: fixtures.run_detail,
      [`/api/runs/${fixtures.run_id}/trace?after=0`]: { events: fixtures.events, next: fixtures.events.length },
    };
    const body = routes[url] ?? (url.includes("/trace?after=") ? { events: [], next: 0 } : null);
    return { ok: !!body, status: body ? 200 : 404, statusText: "",
             json: async () => body, text: async () => JSON.stringify(body) };
  };
  // ページ内スクリプトを実行する（"use strict" 付きの eval では関数宣言が
  // global にならないため、実ブラウザと同じになるよう先頭の指令だけ外す）
  for (const script of window.document.querySelectorAll("script:not([src])")) {
    window.eval(script.textContent.replace(/^\s*"use strict";/, ""));
  }
  const $ = (id) => window.document.getElementById(id);
  const fire = (element, type) => element.dispatchEvent(
    new window.Event(type, { bubbles: true }));

  // 1. 分子のプレビュー
  $("smi").value = fixtures.molecule;
  fire($("smi"), "input");
  await wait(400);
  const molecule = {
    drawn: ($("smiPreview").querySelector("svg") || { childNodes: [] }).childNodes.length,
    status: $("smiStat").textContent.trim(),
  };

  // 2. 反応式のプレビュー
  $("smi").value = fixtures.reaction;
  fire($("smi"), "input");
  await wait(400);
  const previewSvg = $("smiPreview").querySelector("svg");
  const reaction = {
    elements: previewSvg ? previewSvg.querySelectorAll("line,text").length : 0,
    status: $("smiStat").textContent.trim(),
  };

  // 3. テンプレート適用 + プロンプト内 SMILES の検出
  $("smi").value = fixtures.molecule;
  fire($("smi"), "input");
  const chip = [...$("templates").querySelectorAll(".chip")]
    .find((c) => c.textContent.includes(fixtures.template_label));
  fire(chip, "click");
  await wait(600);
  const template = {
    request: $("req").value,
    task_type: $("ttype").value,
    expect: $("expect").value,
    detected: [...$("reqStructures").querySelectorAll("figcaption")]
      .map((f) => f.textContent),
  };

  // 4. プリセットのクリック
  const preset = [...$("presets").querySelectorAll(".chip")]
    .find((c) => c.textContent === fixtures.preset_label);
  fire(preset, "click");
  await wait(400);
  const presetResult = { smiles: $("smi").value, status: $("smiStat").textContent.trim() };

  // 5. ラン詳細の構造カード
  window.selectRun(fixtures.run_id);
  await wait(900);
  const items = [...($("structures") || { querySelectorAll: () => [] })
    .querySelectorAll("li")];
  const card = {
    hidden: $("structureCard") ? $("structureCard").hidden : null,
    count: items.length,
    reactions: items.filter((li) =>
      (li.querySelector(".smiles").textContent || "").includes(">")).length,
    drawn: items.filter((li) => li.querySelector("svg")
      && li.querySelector("svg").childNodes.length > 0).length,
    labels: items.map((li) => li.querySelector(".label")
      ? li.querySelector(".label").textContent : ""),
  };

  const result = { molecule, reaction, template, preset: presetResult, card,
                   jsdom_errors: [...new Set(errors)] };
  window.close();
  return result;
}

(async () => {
  const output = { report: await checkReport(), ui: await checkUi() };
  process.stdout.write(JSON.stringify(output, null, 2));
})().catch((error) => {
  process.stdout.write(JSON.stringify({ fatal: String(error && (error.stack || error)) }));
  process.exit(1);
});
