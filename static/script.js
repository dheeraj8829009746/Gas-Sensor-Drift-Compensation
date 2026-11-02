/* Gas Sensor AI dashboard: charts, tables and live prediction */
const COLORS = ["#4f46e5", "#059669", "#f59e0b", "#dc2626", "#0ea5e9", "#8b5cf6", "#ec4899", "#14b8a6", "#84cc16", "#64748b"];
const charts = {};
const cache = {};
const loaded = {};

const sections = {
    dashboard: "Gas Classification under Sensor Drift",
    eda: "Dataset & EDA",
    classification: "Model Performance",
    selection: "Model Selection",
    crossbatch: "Baseline Cross-Batch Analysis",
    drift: "Sensor Drift Analysis",
    compensation: "Drift Compensation",
    statistics: "Statistical Validation",
    pipeline: "Final Model Pipeline",
    predict: "Gas Prediction",
    about: "About Project"
};

const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "-" : Number(v).toFixed(d);
const pfmt = p => (p === null || p === undefined) ? "-" : (p < 0.001 ? "< 0.001" : Number(p).toFixed(4));
const signed = v => (v > 0 ? "+" : "") + fmt(v);

async function api(url) {
    if (!cache[url]) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(url + " failed (" + r.status + ")");
        cache[url] = await r.json();
    }
    return cache[url];
}

function mk(id, cfg) {
    if (charts[id]) charts[id].destroy();
    cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false }, cfg.options || {});
    charts[id] = new Chart($(id), cfg);
}

function fillTable(id, rows, classify) {
    $(id).innerHTML = rows.map(r =>
        "<tr>" + r.map((c, i) => `<td${classify ? classify(c, i) : ""}>${c}</td>`).join("") + "</tr>").join("");
}

const gainClass = (c, i) => {
    if (typeof c === "string" && (c.startsWith("+") || c.startsWith("-")) && !isNaN(parseFloat(c)))
        return ` class="${c.startsWith("+") ? "pos" : "neg"}"`;
    return "";
};

function heatTable(labels, matrix) {
    let html = "<table class='confusion-table'><tr><th>Actual \\ Pred</th>" + labels.map(l => `<th>${l}</th>`).join("") + "</tr>";
    matrix.forEach((row, i) => {
        const total = row.reduce((a, b) => a + b, 0) || 1;
        html += `<tr><th>${labels[i]}</th>` + row.map((v, j) => {
            const a = v / total;
            const bg = i === j ? `rgba(5,150,105,${0.12 + 0.75 * a})` : `rgba(220,38,38,${Math.min(0.85, a * 2.2)})`;
            return `<td style="background:${bg};color:${a > 0.55 ? '#fff' : '#111827'}">${v}</td>`;
        }).join("") + "</tr>";
    });
    return html + "</table>";
}

function batchLabels(n = 10, from = 1) {
    return Array.from({ length: n }, (_, i) => "B" + (i + from));
}

/* ---------------------------- navigation ---------------------------- */
function showSection(id, button) {
    document.querySelectorAll(".page-section").forEach(s => s.classList.remove("active-section"));
    $(id).classList.add("active-section");
    document.querySelectorAll(".nav-item").forEach(i => i.classList.remove("active"));
    button.classList.add("active");
    $("page-title").textContent = sections[id];
    window.scrollTo(0, 0);
    const loaders = {
        eda: loadEDA, classification: loadModels, selection: loadSelection, crossbatch: loadCrossBatch,
        drift: loadDrift, compensation: loadCompensation, statistics: loadStatistics, pipeline: loadPipeline
    };
    if (loaders[id]) loaders[id]().catch(showError);
}

function showError(e) {
    console.error(e);
}

/* ----------------------------- dashboard ---------------------------- */
async function loadOverview() {
    const d = await api("/api/overview");
    $("total-samples").textContent = d.samples.toLocaleString();
    $("total-sensors").textContent = d.sensors;
    $("total-batches").textContent = d.batches;
    $("total-gases").textContent = d.gas_classes;
    $("hero-model").textContent = d.selected_model;
    $("hero-method").textContent = d.best_method_label;
    $("selected-model").textContent = d.selected_model;
    $("kpi-before").textContent = d.avg_before + "%";
    $("kpi-after").textContent = d.avg_after + "%";
    $("kpi-gain").textContent = signed(d.improvement) + " pp";
    $("kpi-err").textContent = fmt(d.relative_error_reduction, 1) + "%";

    mk("gasChart", {
        type: "bar",
        data: { labels: d.gas_distribution.map(g => g.gas), datasets: [{ label: "Samples", data: d.gas_distribution.map(g => g.count), backgroundColor: COLORS, borderRadius: 8 }] },
        options: { plugins: { legend: { display: false } } }
    });

    const steps = [["Dataset", "UCI Gas Sensor Array Drift, 10 batches"], ["Preprocessing", "Validation, cleaning, scaling"],
        ["Classification", "RF / SVM / XGBoost trained on Batch 1"], ["Model selection", "Best average cross-batch accuracy"],
        ["Drift detection", "Mean, variance, KS, drift score, PCA"], ["Drift compensation", "Align every batch to Batch 1"],
        ["Validation", "Multi-batch, multi-seed, significance tests"]];
    $("workflow").innerHTML = steps.map((s, i) => `<div><span>${String(i + 1).padStart(2, "0")}</span><b>${s[0]}</b><small>${s[1]}</small></div>`).join("");

    const c = await api("/api/compensation");
    baBars("dashBA", c.before_after);
}

function baBars(id, rows) {
    mk(id, {
        type: "bar",
        data: {
            labels: rows.map(r => "Batch " + r.batch),
            datasets: [
                { label: "Before compensation", data: rows.map(r => r.acc_before), backgroundColor: "#f87171", borderRadius: 5 },
                { label: "After compensation", data: rows.map(r => r.acc_after), backgroundColor: "#059669", borderRadius: 5 }]
        },
        options: { scales: { y: { min: 0, max: 100, title: { display: true, text: "Accuracy (%)" } } } }
    });
}

/* -------------------------------- EDA -------------------------------- */
async function loadEDA() {
    const d = await api("/api/eda");
    fillTable("sensor-table", d.sensor_statistics.map(s => [s.sensor, s.mean, s.std, s.min, s.max]));
    const gases = Object.keys(d.batch_gas);
    mk("batchSizeChart", {
        type: "bar",
        data: { labels: batchLabels(), datasets: gases.map((g, i) => ({ label: g, data: d.batch_gas[g], backgroundColor: COLORS[i] })) },
        options: { scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: "Samples" } } } }
    });
    const h = d.histogram.sensor_1;
    mk("sensorHistogram", {
        type: "bar",
        data: { labels: h.bins.slice(0, -1).map(v => v.toFixed(0)), datasets: [{ label: "Frequency", data: h.counts, backgroundColor: "#6366f1", borderRadius: 4 }] },
        options: { plugins: { legend: { display: false } } }
    });
    drawCorrelation(d.correlation);
}

function drawCorrelation(m) {
    const canvas = $("correlationChart");
    const size = Math.min(canvas.parentElement.clientWidth || 380, 380);
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext("2d");
    const cell = size / m.length;
    m.forEach((row, i) => row.forEach((v, j) => {
        const t = (v + 1) / 2;
        ctx.fillStyle = `rgb(${Math.round(255 * t)},${Math.round(80 + 60 * (1 - Math.abs(v)))},${Math.round(255 * (1 - t))})`;
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
    }));
}

/* ---------------------- classification & selection ------------------- */
async function loadModels() {
    const d = await api("/api/models");
    const key = { "Random Forest": "rf", "SVM": "svm", "XGBoost": "xgb" };
    Object.entries(key).forEach(([m, k]) => {
        $(k + "-accuracy").textContent = d.metrics[m].accuracy + "%";
        $(k + "-f1").textContent = d.metrics[m].f1 + "%";
    });
    const names = Object.keys(d.metrics);
    mk("modelChart", {
        type: "bar",
        data: {
            labels: names, datasets: ["accuracy", "precision", "recall", "f1"].map((k, i) =>
                ({ label: k === "f1" ? "Macro F1" : k[0].toUpperCase() + k.slice(1), data: names.map(n => d.metrics[n][k]), backgroundColor: COLORS[i], borderRadius: 6 }))
        },
        options: { scales: { y: { min: 80, max: 100 } } }
    });
    await loadConfusionMatrix();
}

async function loadConfusionMatrix() {
    const d = await api("/api/confusion/" + encodeURIComponent($("confusion-select").value));
    $("confusion-container").innerHTML = heatTable(d.labels, d.matrix);
}

async function loadSelection() {
    const d = await api("/api/models");
    fillTable("model-table", d.selection.map(r => [
        `<b>${r.model}</b>${r.model === d.selected_model ? " ★" : ""}`, r.baseline + "%", r.avg_cross + "%", signed(r.change) + " pp", r.worst + "%"]), gainClass);
    mk("selectionChart", {
        type: "bar",
        data: {
            labels: d.selection.map(r => r.model), datasets: [
                { label: "Batch 1 accuracy", data: d.selection.map(r => r.baseline), backgroundColor: "#6366f1", borderRadius: 6 },
                { label: "Avg cross-batch accuracy", data: d.selection.map(r => r.avg_cross), backgroundColor: "#f59e0b", borderRadius: 6 }]
        },
        options: { scales: { y: { min: 0, max: 100 } } }
    });
}

async function loadCrossBatch() {
    const d = await api("/api/cross-batch");
    const models = ["Random Forest", "SVM", "XGBoost"];
    fillTable("batch-table", d.map(r => [`Batch ${r.batch}`, ...models.map(m => r[m] + "%"), ...models.map(m => signed(r[m + "_change"]) + " pp")]), gainClass);
    mk("batchChart", {
        type: "line",
        data: { labels: d.map(r => "Batch " + r.batch), datasets: models.map((m, i) => ({ label: m, data: d.map(r => r[m]), borderColor: COLORS[i], backgroundColor: COLORS[i], tension: 0.3, pointRadius: 4 })) },
        options: { scales: { y: { min: 0, max: 100, title: { display: true, text: "Accuracy (%)" } } } }
    });
    mk("degradeChart", {
        type: "bar",
        data: { labels: d.map(r => "Batch " + r.batch), datasets: models.map((m, i) => ({ label: m, data: d.map(r => r[m + "_change"]), backgroundColor: COLORS[i] })) },
        options: { scales: { y: { title: { display: true, text: "Change vs Batch 1 (pp)" } } } }
    });
}

/* ------------------------------- drift ------------------------------- */
let driftData = null;

async function loadDrift() {
    const d = driftData = await api("/api/drift");
    const key = "drift_score_" + d.best_method;
    mk("driftScoreChart", {
        type: "line",
        data: {
            labels: d.rows.map(r => "Batch " + r.batch), datasets: [
                { label: "Drift score (raw)", data: d.rows.map(r => r.drift_score), borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,.12)", fill: true, tension: 0.3 },
                { label: "Drift score (compensated)", data: d.rows.map(r => r[key]), borderColor: "#059669", backgroundColor: "rgba(5,150,105,.12)", fill: true, tension: 0.3 }]
        },
        options: { scales: { y: { min: 0, title: { display: true, text: "Mean KS statistic" } } } }
    });
    mk("driftMetricsChart", {
        type: "bar",
        data: {
            labels: d.rows.map(r => "Batch " + r.batch), datasets: [
                { label: "Mean shift (σ)", data: d.rows.map(r => r.mean_shift), backgroundColor: "#6366f1", borderRadius: 4 },
                { label: "Variance shift", data: d.rows.map(r => r.var_shift), backgroundColor: "#f59e0b", borderRadius: 4 }]
        }
    });
    if (!$("sensor-select").options.length) {
        $("sensor-select").innerHTML = Array.from({ length: 128 }, (_, i) => `<option value="${i}">Sensor ${i + 1}</option>`).join("");
        $("dist-batch").innerHTML = Array.from({ length: 9 }, (_, i) => `<option value="${i + 2}"${i === 8 ? " selected" : ""}>Batch ${i + 2}</option>`).join("");
    }
    drawSensorStats();
    await drawDistribution();
    mk("driftAccChart", {
        type: "scatter",
        data: { datasets: [{ label: "Batches 2-10", data: d.vs_accuracy.map(p => ({ x: p.drift, y: p.acc, b: p.batch })), backgroundColor: "#4f46e5", pointRadius: 7 }] },
        options: {
            scales: { x: { title: { display: true, text: "Drift score" } }, y: { title: { display: true, text: "Accuracy (%)" } } },
            plugins: { tooltip: { callbacks: { label: c => `Batch ${c.raw.b}: drift ${c.raw.x}, accuracy ${c.raw.y}%` } } }
        }
    });
    const c = d.correlation;
    $("drift-corr").innerHTML = `Pearson r = <b>${fmt(c.pearson_r, 3)}</b> (p ${pfmt(c.pearson_p)}), Spearman ρ = <b>${fmt(c.spearman_rho, 3)}</b> (p ${pfmt(c.spearman_p)}). A strongly negative value means batches with a higher drift score are classified less accurately.`;
    fillTable("drift-table", d.rows.map(r => [`Batch ${r.batch}`, fmt(r.mean_shift, 3), fmt(r.var_shift, 3), fmt(r.ks_pooled, 3), fmt(r.drift_score, 3), fmt(r[key], 3)]));
    $("pca-var").textContent = "PC1+PC2 explain " + fmt((d.pca.before.explained[0] + d.pca.before.explained[1]) * 100, 1) + "% variance (raw)";
    pcaChart("pcaBefore", d.pca.before.points);
    pcaChart("pcaAfter", d.pca.after.points);
}

function drawSensorStats() {
    const j = +$("sensor-select").value;
    const mean = driftData.sensor_mean.map(r => r[j]);
    const std = driftData.sensor_std.map(r => r[j]);
    mk("sensorStatsChart", {
        type: "line",
        data: {
            labels: batchLabels(), datasets: [
                { label: "Mean", data: mean, borderColor: "#4f46e5", backgroundColor: "#4f46e5", tension: 0.3, pointRadius: 4 },
                { label: "Mean + Std", data: mean.map((m, i) => m + std[i]), borderColor: "#a5b4fc", borderDash: [5, 4], pointRadius: 0 },
                { label: "Mean - Std", data: mean.map((m, i) => m - std[i]), borderColor: "#a5b4fc", borderDash: [5, 4], pointRadius: 0 }]
        }
    });
}

async function drawDistribution() {
    const s = $("dist-sensor").value, b = $("dist-batch").value;
    const d = await api(`/api/drift/distribution?sensor=${s}&batch=${b}`);
    [["distRaw", d.raw], ["distComp", d.comp]].forEach(([id, h]) => {
        const mid = h.edges.slice(0, -1).map((e, i) => ((e + h.edges[i + 1]) / 2).toPrecision(3));
        mk(id, {
            type: "line",
            data: {
                labels: mid, datasets: [
                    { label: "Batch 1", data: h.ref, borderColor: "#4f46e5", backgroundColor: "rgba(79,70,229,.25)", fill: true, tension: 0.35, pointRadius: 0 },
                    { label: "Batch " + b, data: h.cur, borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,.25)", fill: true, tension: 0.35, pointRadius: 0 }]
            },
            options: { scales: { y: { title: { display: true, text: "% of samples" } } } }
        });
    });
}

function pcaChart(id, points) {
    const batches = [...new Set(points.map(p => p.batch))];
    mk(id, {
        type: "scatter",
        data: { datasets: batches.map((b, i) => ({ label: "Batch " + b, data: points.filter(p => p.batch === b).map(p => ({ x: p.x, y: p.y })), backgroundColor: COLORS[i] + "AA", pointRadius: 3 })) },
        options: { scales: { x: { title: { display: true, text: "PC1" } }, y: { title: { display: true, text: "PC2" } } } }
    });
}

/* ---------------------------- compensation --------------------------- */
async function loadCompensation() {
    const c = await api("/api/compensation");
    $("best-method").textContent = c.best_method_label;
    $("best-method-note").textContent = `Chosen by highest average accuracy on batches 2-10 with ${c.selected_model}. Average ${c.avg_before}% → ${c.avg_after}%.`;
    $("best-gain").textContent = signed(c.improvement) + " pp";

    mk("methodChart", {
        type: "line",
        data: { labels: batchLabels(), datasets: c.methods.map((m, i) => ({ label: m.label, data: m.per_batch, borderColor: COLORS[i], backgroundColor: COLORS[i], tension: 0.25, pointRadius: 3, borderDash: m.method === "none" ? [6, 4] : [] })) },
        options: { scales: { y: { title: { display: true, text: "Accuracy (%)" } } } }
    });
    fillTable("method-table", c.methods.map(m => [m.method === c.best_method ? `<b>${m.label}</b> ★` : m.label, m.avg_selected + "%", m.avg_all["Random Forest"] + "%", m.avg_all["SVM"] + "%", m.avg_all["XGBoost"] + "%"]));

    baBars("baChart", c.before_after);
    mk("gainChart", {
        type: "bar",
        data: { labels: c.before_after.slice(1).map(r => "Batch " + r.batch), datasets: [{ label: "Accuracy gain (pp)", data: c.before_after.slice(1).map(r => r.improvement), backgroundColor: c.before_after.slice(1).map(r => r.improvement >= 0 ? "#059669" : "#dc2626"), borderRadius: 5 }] },
        options: { plugins: { legend: { display: false } } }
    });
    fillTable("ba-table", c.before_after.map(r => [`Batch ${r.batch}${r.batch === 1 ? " (hold-out)" : ""}`, r.acc_before + "%", r.acc_after + "%", signed(r.improvement) + " pp", r.f1_before + "%", r.f1_after + "%"]), gainClass);

    const names = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1-score"]];
    mk("prfChart", {
        type: "bar",
        data: {
            labels: names.map(n => n[1]), datasets: [
                { label: "Before", data: names.map(n => c.prf.before[n[0]]), backgroundColor: "#f87171", borderRadius: 6 },
                { label: "After", data: names.map(n => c.prf.after[n[0]]), backgroundColor: "#059669", borderRadius: 6 }]
        },
        options: { scales: { y: { min: 0, max: 100 } } }
    });
    fillTable("gas-table", c.per_gas.map(g => [g.gas, g.recall_before + "%", g.recall_after + "%", g.f1_before + "%", g.f1_after + "%"]));

    const models = Object.keys(c.all_models);
    const avg = a => a.slice(1).reduce((x, y) => x + y, 0) / 9;
    mk("allModelsChart", {
        type: "bar",
        data: {
            labels: models, datasets: [
                { label: "Before", data: models.map(m => +avg(c.all_models[m].before).toFixed(2)), backgroundColor: "#f87171", borderRadius: 6 },
                { label: "After", data: models.map(m => +avg(c.all_models[m].after).toFixed(2)), backgroundColor: "#059669", borderRadius: 6 }]
        },
        options: { scales: { y: { min: 0, max: 100 } } }
    });
    $("cm-before").innerHTML = heatTable(c.labels, c.confusion_before);
    $("cm-after").innerHTML = heatTable(c.labels, c.confusion_after);
}

/* ----------------------------- statistics ---------------------------- */
async function loadStatistics() {
    const s = await api("/api/statistics");
    const b = s.batch_level, sd = s.seeds;
    $("st-before").textContent = `${fmt(b.mean_before)} ± ${fmt(b.std_before)}`;
    $("st-after").textContent = `${fmt(b.mean_after)} ± ${fmt(b.std_after)}`;
    $("st-t").textContent = pfmt(b.p_ttest);
    $("st-w").textContent = pfmt(b.p_wilcoxon);

    mk("seedChart", {
        type: "bar",
        data: {
            labels: batchLabels(9, 2), datasets: [
                { label: "Before (mean ± std)", data: sd.per_batch_before_mean, backgroundColor: "#f87171", borderRadius: 4 },
                { label: "After (mean ± std)", data: sd.per_batch_after_mean, backgroundColor: "#059669", borderRadius: 4 }]
        },
        options: { scales: { y: { min: 0, max: 105 } } },
        plugins: [{
            id: "errbars", afterDatasetsDraw(chart) {
                const ctx = chart.ctx, stds = [sd.per_batch_before_std, sd.per_batch_after_std];
                chart.data.datasets.forEach((ds, di) => chart.getDatasetMeta(di).data.forEach((bar, i) => {
                    const y1 = chart.scales.y.getPixelForValue(ds.data[i] + stds[di][i]);
                    const y2 = chart.scales.y.getPixelForValue(Math.max(0, ds.data[i] - stds[di][i]));
                    ctx.save(); ctx.strokeStyle = "#111827"; ctx.lineWidth = 1.2; ctx.beginPath();
                    ctx.moveTo(bar.x, y1); ctx.lineTo(bar.x, y2); ctx.moveTo(bar.x - 3, y1); ctx.lineTo(bar.x + 3, y1);
                    ctx.moveTo(bar.x - 3, y2); ctx.lineTo(bar.x + 3, y2); ctx.stroke(); ctx.restore();
                }));
            }
        }]
    });

    const mc = s.mcnemar, bs = s.bootstrap, dv = s.drift_vs_accuracy;
    fillTable("stat-table", [
        ["Model / technique", `${s.selected_model} / ${s.best_method_label}`],
        ["Batch accuracy before (mean ± std, n=9)", `${fmt(b.mean_before)} ± ${fmt(b.std_before)} %`],
        ["Batch accuracy after (mean ± std, n=9)", `${fmt(b.mean_after)} ± ${fmt(b.std_after)} %`],
        ["Mean improvement ± std", `${signed(b.mean_improvement)} ± ${fmt(b.std_improvement)} pp`],
        ["95% CI of improvement (t-distribution)", `[${fmt(b.ci_low)}, ${fmt(b.ci_high)}] pp`],
        ["Paired t-test (t, p)", `${fmt(b.t_stat, 3)}, ${pfmt(b.p_ttest)}`],
        ["Wilcoxon signed-rank (p)", pfmt(b.p_wilcoxon)],
        ["Effect size (Cohen's dz)", fmt(b.cohen_dz, 3)],
        ["Batches improved", `${b.batches_improved} / ${b.n}`],
        [`${sd.seeds.length} random seeds: before`, `${fmt(sd.before_mean)} ± ${fmt(sd.before_std)} %`],
        [`${sd.seeds.length} random seeds: after`, `${fmt(sd.after_mean)} ± ${fmt(sd.after_std)} %`],
        ["Seed-level paired t-test p", pfmt(sd.test.p_ttest)],
        [`McNemar exact test (${mc.samples.toLocaleString()} samples)`, `fixed ${mc.fixed_by_compensation}, broken ${mc.broken_by_compensation}, p ${pfmt(mc.p_value)}`],
        ["Bootstrap 95% CI of gain (1000 resamples)", `${signed(bs.mean)} pp [${fmt(bs.ci_low)}, ${fmt(bs.ci_high)}]`],
        ["Drift score vs accuracy (Pearson r)", `${fmt(dv.pearson_r, 3)} (p ${pfmt(dv.pearson_p)})`]
    ]);

    fillTable("stat-methods", s.methods.map(m => [m.method, signed(m.mean_improvement) + " pp", `[${fmt(m.ci_low)}, ${fmt(m.ci_high)}]`, pfmt(m.p_ttest), pfmt(m.p_wilcoxon), fmt(m.cohen_dz, 2), `${m.batches_improved} / ${m.n}`]));

    const sig = (p) => p < 0.05 ? "statistically significant" : "not statistically significant";
    $("stat-interpret").innerHTML = `<b>Interpretation.</b> Across batches 2-10 the average accuracy changed from ${fmt(b.mean_before)}% to ${fmt(b.mean_after)}% (${signed(b.mean_improvement)} percentage points; ${b.batches_improved} of ${b.n} batches improved). The paired t-test is ${sig(b.p_ttest)} (p = ${pfmt(b.p_ttest)}) and the Wilcoxon test is ${sig(b.p_wilcoxon)} (p = ${pfmt(b.p_wilcoxon)}). With only nine batches, the t-test is sensitive to the very uneven size of the gains, so the sample-level McNemar test and the bootstrap interval (which use every test sample) are the stronger evidence: compensation fixed ${mc.fixed_by_compensation} predictions and broke ${mc.broken_by_compensation} (p ${pfmt(mc.p_value)}). Results are stable across random seeds (std ${fmt(sd.after_std)} pp after compensation).`;
}

/* ------------------------------ pipeline ------------------------------ */
let finalRows = [];

async function loadPipeline() {
    const p = await api("/api/pipeline");
    const steps = [["Input", "128 sensor readings + the batch they belong to"],
        ["Preprocessing", "Type and range validation, finite-value check"],
        ["Drift compensation", p.method_label + " (batch statistics from unlabelled samples)"],
        ["Selected model", p.selected_model + " (after StandardScaler)"],
        ["Gas class", "Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone or Toluene"]];
    $("flow").innerHTML = steps.map((s, i) => `<div class="step"><div class="n">${i + 1}</div><b>${s[0]}</b><small>${s[1]}</small></div>` + (i < steps.length - 1 ? '<div class="arrow">→</div>' : "")).join("");
    finalRows = p.final_table;
    fillTable("final-table", p.final_table.map(r => [`Batch ${r.batch}`, fmt(r.drift_before, 3), fmt(r.drift_after, 3), r.acc_before + "%", r.acc_after + "%", signed(r.improvement) + " pp", r.f1_before + "%", r.f1_after + "%"]), gainClass);
}

function downloadCSV() {
    const head = ["batch", "drift_before", "drift_after", "acc_before", "acc_after", "improvement_pp", "f1_before", "f1_after"];
    const lines = [head.join(",")].concat(finalRows.map(r => [r.batch, r.drift_before, r.drift_after, r.acc_before, r.acc_after, r.improvement, r.f1_before, r.f1_after].join(",")));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = "final_results.csv";
    a.click();
}

/* ------------------------------ prediction ----------------------------- */
function buildInputs() {
    $("sensor-inputs").innerHTML = Array.from({ length: 128 }, (_, i) =>
        `<div class="sensor-input"><label>Sensor ${i + 1}</label><input type="number" step="any" id="sensor-${i + 1}"></div>`).join("");
    $("batch-select").innerHTML = '<option value="">No compensation</option>' +
        Array.from({ length: 10 }, (_, i) => `<option value="${i + 1}">Batch ${i + 1}</option>`).join("");
}

function clearInputs() {
    for (let i = 1; i <= 128; i++) $("sensor-" + i).value = "";
    $("true-label").textContent = "";
    $("predict-error").textContent = "";
}

async function loadSample() {
    let b = $("batch-select").value;
    if (!b) { b = String(2 + Math.floor(Math.random() * 9)); $("batch-select").value = b; }
    const r = await fetch("/api/sample?batch=" + b);
    const d = await r.json();
    d.sensors.forEach((v, i) => { $("sensor-" + (i + 1)).value = v; });
    $("true-label").textContent = `Real sample from Batch ${d.batch}. True gas: ${d.true_gas}.`;
    window.lastTrue = d.true_gas;
}

async function predictGas() {
    $("predict-error").textContent = "";
    const sensors = [];
    for (let i = 1; i <= 128; i++) {
        const v = $("sensor-" + i).value;
        if (v === "") { $("predict-error").textContent = `Sensor ${i} is empty. Use "Load real sample" or fill all 128 values.`; return; }
        sensors.push(parseFloat(v));
    }
    const r = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensors, batch: $("batch-select").value || null })
    });
    const d = await r.json();
    if (!r.ok) { $("predict-error").textContent = d.error || "Prediction failed"; return; }
    $("prediction-icon").textContent = d.prediction[0];
    $("prediction-gas").textContent = d.prediction;
    $("prediction-model").textContent = `${d.model} | ${d.compensation}`;
    $("prediction-confidence").textContent = d.confidence === null ? "N/A" : d.confidence + "%";
    const raw = d.uncompensated;
    $("prediction-raw").textContent = `${raw.gas}${raw.confidence === null ? "" : " (" + raw.confidence + "%)"}`;
    const probs = d.probabilities || {};
    $("prob-bars").innerHTML = Object.entries(probs).sort((a, b) => b[1] - a[1]).map(([g, p]) =>
        `<div class="prob-row"><span>${g}</span><i style="width:${Math.max(p, 1)}%"></i><span>${p}%</span></div>`).join("");
    const t = $("true-label");
    if (window.lastTrue) {
        const ok = window.lastTrue === d.prediction;
        t.innerHTML = t.textContent.split(" True gas:")[0] + ` True gas: <b>${window.lastTrue}</b> (<span class="${ok ? "pos" : "neg"}">${ok ? "correct" : "incorrect"}</span>)`;
    }
}

/* -------------------------------- init -------------------------------- */
window.addEventListener("DOMContentLoaded", () => {
    buildInputs();
    loadOverview().catch(showError);
});
