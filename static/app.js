"use strict";

// 可复原示例：一口 f0≈100 的钟（泛音 3,4,5,6,8,9,10）+ 1 条杂峰 668。
const SAMPLE_RECOVER = {
  frequencies: [300.0, 400.5, 499.2, 600.3, 668.0, 800.8, 901.5, 1002.0],
  tolerances:  [1.5,   1.5,   1.5,   1.5,   0.8,   1.5,   1.5,   1.5],
  f0_min: "95", f0_max: "105", max_harmonic: "12", max_rejected: "2",
};

// 无解残音：300/400/500/600 可由 f0≈100 联合解释；668 的基频带（h=7 时
// f₀≈95.4）与联合交集互不相容；846 在候选区间内无任何泛音归属。最多只允许
// 剔除 1 条，而至少要排除两条冲突峰，故不存在共同基频。
const SAMPLE_UNSOLVABLE = {
  frequencies: [300, 400, 500, 600, 668, 846],
  tolerances:  [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
  f0_min: "95", f0_max: "105", max_harmonic: "12", max_rejected: "1",
};

const MIN_ROWS = 6, MAX_ROWS = 18;

const $ = (sel) => document.querySelector(sel);
const peakRows = $("#peakRows");

function makeRow(freq = "", tol = "") {
  const tr = document.createElement("tr");
  const idx = document.createElement("td");
  idx.className = "row-idx";
  const fTd = document.createElement("td");
  const fIn = document.createElement("input");
  fIn.type = "text";
  fIn.inputMode = "decimal";
  fIn.className = "in-freq";
  fIn.value = freq;
  fIn.placeholder = "如 440.00";
  const tTd = document.createElement("td");
  const tIn = document.createElement("input");
  tIn.type = "text";
  tIn.inputMode = "decimal";
  tIn.className = "in-tol";
  tIn.value = tol;
  tIn.placeholder = "如 1.5";
  const dTd = document.createElement("td");
  dTd.className = "drop-col";
  fTd.appendChild(fIn);
  tTd.appendChild(tIn);
  tr.append(idx, fTd, tTd, dTd);
  return tr;
}

function refreshIndex() {
  [...peakRows.children].forEach((tr, i) => {
    tr.querySelector(".row-idx").textContent = i + 1;
  });
  $("#rowCount").textContent = `共 ${peakRows.children.length} 条（允许 ${MIN_ROWS}–${MAX_ROWS} 条）`;
  $("#addRow").disabled = peakRows.children.length >= MAX_ROWS;
  $("#removeRow").disabled = peakRows.children.length <= MIN_ROWS;
}

function setRows(freqs, tols) {
  peakRows.innerHTML = "";
  freqs.forEach((f, i) => peakRows.appendChild(makeRow(f, tols[i])));
  refreshIndex();
}

function loadSample(sample) {
  setRows(sample.frequencies, sample.tolerances);
  $("#f0_min").value = sample.f0_min;
  $("#f0_max").value = sample.f0_max;
  $("#max_harmonic").value = sample.max_harmonic;
  $("#max_rejected").value = sample.max_rejected;
  hideResult();
}

$("#addRow").addEventListener("click", () => {
  if (peakRows.children.length < MAX_ROWS) {
    peakRows.appendChild(makeRow());
    refreshIndex();
  }
});
$("#removeRow").addEventListener("click", () => {
  if (peakRows.children.length > MIN_ROWS) {
    peakRows.lastElementChild.remove();
    refreshIndex();
  }
});
$("#sampleRecover").addEventListener("click", () => loadSample(SAMPLE_RECOVER));
$("#sampleUnsolvable").addEventListener("click", () => loadSample(SAMPLE_UNSOLVABLE));

function fmt(x, digits = 6) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  const abs = Math.abs(x);
  if (abs >= 1e6 || abs < 1e-4) return x.toExponential(4);
  return String(parseFloat(x.toPrecision(digits)));
}
function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (x * 100).toPrecision(4) + " %";
}

function hideResult() {
  $("#resultCard").classList.add("hidden");
  $("#inputError").classList.add("hidden");
}

function showError(msg) {
  const box = $("#inputError");
  box.textContent = msg;
  box.classList.remove("hidden");
  $("#resultCard").classList.add("hidden");
}

$("#solveBtn").addEventListener("click", async () => {
  const frequencies = [], tolerances = [];
  for (const tr of peakRows.children) {
    frequencies.push(tr.querySelector(".in-freq").value.trim());
    tolerances.push(tr.querySelector(".in-tol").value.trim());
  }
  const body = {
    frequencies,
    tolerances,
    f0_min: $("#f0_min").value.trim(),
    f0_max: $("#f0_max").value.trim(),
    max_harmonic: $("#max_harmonic").value.trim(),
    max_rejected: $("#max_rejected").value.trim(),
  };
  const btn = $("#solveBtn");
  btn.disabled = true;
  btn.textContent = "联合搜索中…";
  try {
    const resp = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok || data.status !== "ok") {
      showError(data.message || `请求失败（HTTP ${resp.status}）`);
      return;
    }
    render(data, body);
  } catch (err) {
    showError(`无法连接服务端：${err}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "联合求解";
  }
});

function render(data, rawInput) {
  $("#inputError").classList.add("hidden");
  const card = $("#resultCard");
  card.classList.remove("hidden");

  const sol = data.solution;
  const K = sol.adopted_count;
  const n = sol.peaks.length;
  const hasF0 = Number.isFinite(sol.f0);

  // 结论横幅
  const verdict = $("#verdict");
  let banner;
  if (K === n) {
    banner = `<div class="verdict-banner verdict-ok"><span class="big">✓</span>
      <span>全部 <b>${n}</b> 条峰可由同一口钟解释，统一基频
      <b>f₀ = ${fmt(sol.f0, 8)}</b>，无需剔除杂峰。</span></div>`;
  } else if (hasF0) {
    banner = `<div class="verdict-banner verdict-partial"><span class="big">◑</span>
      <span>在剔除上限内，<b>${K}</b> 条峰可由统一基频
      <b>f₀ = ${fmt(sol.f0, 8)}</b> 联合解释，
      <b>${sol.rejected_count}</b> 条峰被剔除；原始峰值完整保留于下表。</span></div>`;
  } else {
    banner = `<div class="verdict-banner verdict-none"><span class="big">✕</span>
      <span><b>无解：</b>在允许剔除 ${data.input.max_rejected} 条杂峰的条件下，
      不存在任何能联合解释剩余峰的共同基频。下表保留全部原始峰值。</span></div>`;
  }
  verdict.innerHTML = banner;

  // 证据
  const evBox = $("#evidence");
  if (data.evidence) {
    const ev = data.evidence;
    const others = ev.excluded_indices.filter((i) => i !== ev.index).map((i) => i + 1);
    let bandsText;
    if (ev.valid_bands.length === 0) {
      bandsText = `该峰在候选基频闭区间内对任何 1…${data.input.max_harmonic} 号泛音都不落容差，`;
    } else {
      const bands = ev.valid_bands
        .map((b) => `h=${b.harmonic} 时 f₀∈[${fmt(b.f0_low)}, ${fmt(b.f0_high)}]`)
        .join("；");
      bandsText = `该峰单独允许的基频带为：${bands}；这些带与最优联合解释的公共交集互不相容，`;
    }
    evBox.innerHTML =
      `<strong>⚠ 按峰值顺序最早无法纳入的证据：第 ${ev.index + 1} 条峰（${fmt(ev.frequency)} ± ${fmt(ev.tolerance)}）。</strong><br>` +
      bandsText +
      `全局最多只能找到 ${ev.optimal_count} 条峰共享同一基频` +
      (data.feasible
        ? "。"
        : `，而剔除预算要求至少联合容纳 <b>${ev.required_count}</b> 条，故不存在满足条件的共同基频。`) +
      ` 强制纳入该峰后，在剔除上限内最多只能联合容纳 ` +
      `<b>${ev.max_count_with_peak}</b> 条。` +
      (others.length ? ` 同样被所有最优联合解释排除的还有第 ${others.join("、")} 条峰。` : "");
    evBox.classList.remove("hidden");
  } else {
    evBox.classList.add("hidden");
  }

  // 指标
  const interval = hasF0
    ? `[${fmt(sol.f0_interval[0], 8)}, ${fmt(sol.f0_interval[1], 8)}]`
    : "—";
  $("#stats").innerHTML = `
    <div class="stat"><div class="k">统一基频 f₀</div><div class="v">${hasF0 ? fmt(sol.f0, 8) : "无共同基频"}</div></div>
    <div class="stat"><div class="k">可行 f₀ 闭区间交集</div><div class="v">${interval}</div></div>
    <div class="stat"><div class="k">采用峰 / 剔除峰</div><div class="v">${K} / ${sol.rejected_count}</div></div>
    <div class="stat"><div class="k">最大相对误差</div><div class="v">${fmtPct(sol.max_rel_error)}</div></div>
    <div class="stat"><div class="k">误差平方和 SSE</div><div class="v">${fmt(sol.sse, 8)}</div></div>
  `;

  // 逐峰表（保留原始录入值）
  const rows = $("#resultRows");
  rows.innerHTML = "";
  sol.peaks.forEach((p) => {
    const tr = document.createElement("tr");
    tr.className = p.adopted ? "row-adopted" : "row-rejected";
    const isEarliest = data.evidence && data.evidence.index === p.index;
    const td = (text, cls = "") => {
      const c = document.createElement("td");
      if (cls) c.className = cls;
      c.innerHTML = text;
      return c;
    };
    tr.appendChild(td(String(p.index + 1)));
    tr.appendChild(td(`${fmt(p.frequency, 8)}${isEarliest ? '<span class="earliest-mark">最早证据</span>' : ""}`));
    tr.appendChild(td(fmt(p.tolerance)));
    if (p.adopted) {
      tr.appendChild(td('<span class="tag tag-adopt">采用</span>'));
      tr.appendChild(td(String(p.harmonic), "cell-h"));
      tr.appendChild(td(fmt(p.predicted, 8)));
      tr.appendChild(td(fmt(p.residual, 8)));
      tr.appendChild(td(fmt(p.abs_error, 8)));
      tr.appendChild(td(fmtPct(p.rel_error)));
      tr.appendChild(td(p.within_tolerance ? '<span class="tag-yes">✓ 在容差内</span>'
                                           : '<span class="tag-no">✗ 超出容差</span>'));
    } else {
      const tag = data.feasible
        ? '<span class="tag tag-reject">剔除（杂峰）</span>'
        : '<span class="tag tag-reject">无共同基频可纳入</span>';
      tr.appendChild(td(tag));
      tr.appendChild(td("—", "cell-na"));
      for (let k = 0; k < 5; k++) tr.appendChild(td("—", "cell-na"));
    }
    rows.appendChild(tr);
  });

  $("#methodNote").textContent =
    "选择依据（按序）：采用峰数最多 → 最大相对误差最小 → 误差平方和最小 → 按录入顺序最早的泛音序号方案；" +
    "所有采用峰的容差区间在同一 f₀ 下交集非空，序号严格递增且不重复。";

  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// 初始空表（6 条）
setRows(Array(MIN_ROWS).fill(""), Array(MIN_ROWS).fill(""));
