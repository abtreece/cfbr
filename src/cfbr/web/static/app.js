"use strict";

// -- filters: selects submit their form on change --------------------------------
document.querySelectorAll("select[data-autosubmit]").forEach((sel) => {
  sel.addEventListener("change", () => sel.form.requestSubmit());
});

// -- row filters: <select data-row-filter="tableId" data-attr="pos"> hides rows whose data-pos differs --
document.querySelectorAll("select[data-row-filter]").forEach((sel) => {
  sel.addEventListener("change", () => {
    const id = sel.dataset.rowFilter;
    const active = Array.from(document.querySelectorAll(`select[data-row-filter="${id}"]`))
      .filter((s) => s.value).map((s) => [s.dataset.attr, s.value]);
    document.querySelectorAll(`#${id} tbody tr`).forEach((tr) => {
      tr.hidden = !active.every(([attr, value]) => tr.dataset[attr] === value);
    });
  });
});

// -- Class toggles: <button data-toggle-class="more" data-target="tableId"> toggles a class,
//    remembered per table in this browser. Labels default to "More/Fewer columns"; a button
//    may name its own with data-label-on / data-label-off ("Show 2 voided entries"). -------
document.querySelectorAll("button[data-toggle-class]").forEach((btn) => {
  const target = document.getElementById(btn.dataset.target);
  if (!target) return;
  const cls = btn.dataset.toggleClass, key = `cfbr-${btn.dataset.target}-${cls}`;
  const labelOn = btn.dataset.labelOn || "Fewer columns", labelOff = btn.dataset.labelOff || "More columns";
  const set = (on) => {
    target.classList.toggle(cls, on);
    btn.setAttribute("aria-pressed", String(on));
    btn.textContent = on ? labelOn : labelOff;
  };
  try { set(localStorage.getItem(key) === "1"); } catch (e) { /* storage unavailable */ }
  btn.addEventListener("click", () => {
    const on = !target.classList.contains(cls);
    set(on);
    try { localStorage.setItem(key, on ? "1" : "0"); } catch (e) { /* storage unavailable */ }
  });
});

// -- theme toggle --------------------------------------------------------------------
document.getElementById("theme-toggle")?.addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("cfbr-theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
  document.querySelectorAll("figure.chart").forEach((f) => f._render && f._render());
});

// -- links to a collapsed <details> (e.g. "How it works ↑") open it ------------------
function openDetails(hash) {
  const el = hash && document.getElementById(hash.slice(1));
  if (el && el.tagName === "DETAILS") el.open = true;
}
window.addEventListener("hashchange", () => openDetails(location.hash));
// hashchange doesn't fire when the link's fragment is already the current one.
document.addEventListener("click", (e) => {
  const a = e.target.closest('a[href*="#"]');
  if (a && a.pathname === location.pathname) openDetails(a.hash);
});
openDetails(location.hash);

// -- sortable tables ------------------------------------------------------------------
document.querySelectorAll("table.sortable").forEach((table) => {
  const headers = Array.from(table.querySelectorAll("th"));
  headers.forEach((th, col) => {
    if (!th.dataset.sort) return;
    th.tabIndex = 0;
    const sort = () => {
      const asc = th.getAttribute("aria-sort") === "descending" ||
        (!th.hasAttribute("aria-sort") && (th.dataset.sort === "text" || th.dataset.first === "asc"));
      headers.forEach((h) => h.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", asc ? "ascending" : "descending");
      const body = table.tBodies[0];
      const rows = Array.from(body.rows);
      const key = (r) => {
        const cell = r.cells[col];
        if (th.dataset.sort === "num") return parseFloat(cell.dataset.v ?? cell.textContent);
        return cell.textContent.trim().toLowerCase();
      };
      rows.sort((a, b) => {
        const x = key(a), y = key(b);
        const c = typeof x === "number" ? x - y : x.localeCompare(y);
        return asc ? c : -c;
      });
      rows.forEach((r) => body.appendChild(r));
    };
    th.addEventListener("click", sort);
    th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } });
  });
});

// -- line charts ------------------------------------------------------------------------
const SVG = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const fmt = (v) => (v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1));
const fmtPlain = (v) => (v == null ? "—" : v.toFixed(2));
const fmtDollars = (v) => (v == null ? "—" : (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 2 }));
const shortX = (s) => s.replace(/^Week /, "").replace(/^Preseason$/, "Pre").replace(/^Postseason /, "P");

function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let t = Math.floor(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

function lineChart(fig) {
  const spec = JSON.parse(fig.querySelector("script[type='application/json']").textContent);
  const body = fig.querySelector(".chart-body");
  const signedValues = spec.signed !== false;
  const dollars = spec.format === "dollars";
  const f = dollars ? fmtDollars : signedValues ? fmt : fmtPlain;
  const n = spec.x.length;
  let active = null;

  // Legend (always present for >= 2 series; line keys mirror the marks).
  const legend = document.createElement("div");
  legend.className = "legend";
  if (spec.series.length > 1) spec.series.forEach((s, i) => {
    const item = document.createElement("span");
    const key = document.createElement("i");
    key.className = "key";
    key.style.background = `var(--series-${i + 1})`;
    item.appendChild(key);
    item.appendChild(document.createTextNode(s.name));
    legend.appendChild(item);
  });
  if (legend.childElementCount) fig.insertBefore(legend, body);

  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  fig.appendChild(tip);

  function render() {
    body.textContent = "";
    const W = Math.max(body.clientWidth, 280), H = 240;
    const m = { l: 40, r: 16, t: 10, b: 24 };
    const all = spec.series.flatMap((s) => s.values).filter((v) => v != null);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (signedValues) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); } else { lo = Math.min(lo, Math.floor(lo)); }
    const ticks = niceTicks(lo, hi);
    lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo || 1)) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": fig.getAttribute("aria-label") }, body);

    // End labels: shown only when they don't collide vertically and fit in at most 35% of the
    // width. The right margin is sized to the widest label so labels never clip at the card
    // edge; otherwise the legend and tooltip carry identity.
    const ends = spec.series.map((s, si) => {
      let i = s.values.length - 1;
      while (i >= 0 && s.values[i] == null) i--;
      const text = i < 0 ? "" : spec.series.length > 1 ? `${s.name} ${f(s.values[i])}` : f(s.values[i]);
      return { si, i, v: s.values[i], yy: i >= 0 ? y(s.values[i]) : null, text };
    }).filter((e) => e.i >= 0);
    const sorted = [...ends].sort((a, b) => a.yy - b.yy);
    let showLabels = sorted.every((e, k) => k === 0 || e.yy - sorted[k - 1].yy >= 13);
    if (showLabels && ends.length) {
      const probe = el("text", { class: "endlabel", x: 0, y: -100 }, svg);
      const widest = Math.max(...ends.map((e) => { probe.textContent = e.text; return probe.getComputedTextLength(); }));
      probe.remove();
      const need = Math.ceil(widest) + 14;
      if (need <= W * 0.35) m.r = Math.max(m.r, need); else showLabels = false;
    }
    const x = (i) => m.l + (n === 1 ? (W - m.l - m.r) / 2 : (i * (W - m.l - m.r)) / (n - 1));

    ticks.forEach((t) => {
      el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 && signedValues ? "zero" : "gridline" }, svg);
      const lbl = el("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "tick" }, svg);
      lbl.textContent = dollars ? fmtDollars(t) : signedValues ? fmt(t).replace(".0", "") : String(t);
    });
    const every = Math.max(1, Math.ceil(n / Math.floor((W - m.l - m.r) / 44)));
    const lastGrid = Math.floor((n - 1) / every) * every;
    spec.x.forEach((lab, i) => {
      if (i % every && i !== n - 1) return;
      // The last point is always labeled; drop the grid label just before it if they would touch.
      if (i === lastGrid && i !== n - 1 && x(n - 1) - x(i) < 40) return;
      const t = el("text", { x: x(i), y: H - 6, "text-anchor": "middle", class: "tick" }, svg);
      t.textContent = shortX(lab);
    });

    spec.series.forEach((s, si) => {
      const color = `var(--series-${si + 1})`;
      const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
      const path = el("polyline", { points: pts.map((p) => p.join(",")).join(" "), class: "series" }, svg);
      path.style.stroke = color;
    });

    ends.forEach((e) => {
      const dot = el("circle", { cx: x(e.i), cy: e.yy, r: 4, class: "dot" }, svg);
      dot.style.fill = `var(--series-${e.si + 1})`;
      if (showLabels) {
        const t = el("text", { x: x(e.i) + 8, y: e.yy + 4, class: "endlabel" }, svg);
        t.textContent = e.text;
      }
    });

    const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
    const hoverDots = spec.series.map((_, si) => {
      const d = el("circle", { r: 4, class: "dot", visibility: "hidden" }, svg);
      d.style.fill = `var(--series-${si + 1})`;
      return d;
    });
    const hit = el("rect", { x: m.l - 12, y: 0, width: W - m.l - m.r + 24, height: H, fill: "transparent" }, svg);

    const show = (i) => {
      active = i;
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      spec.series.forEach((s, si) => {
        const v = s.values[i];
        const d = hoverDots[si];
        if (v == null) { d.setAttribute("visibility", "hidden"); return; }
        d.setAttribute("cx", x(i)); d.setAttribute("cy", y(v)); d.setAttribute("visibility", "visible");
      });
      tip.textContent = "";
      const head = document.createElement("div");
      head.className = "tt-head";
      head.textContent = spec.x[i] + (spec.n ? ` · ${spec.n[i]} games` : "");
      tip.appendChild(head);
      spec.series.forEach((s, si) => {
        const row = document.createElement("div");
        row.className = "tt-row";
        const key = document.createElement("i");
        key.className = "key";
        key.style.background = `var(--series-${si + 1})`;
        const b = document.createElement("b");
        b.textContent = f(s.values[i]);
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = s.name;
        row.append(key, b, name);
        tip.appendChild(row);
      });
      tip.hidden = false;
      const scale = body.clientWidth / W;
      const px = x(i) * scale + body.offsetLeft;
      const left = px + 12 + tip.offsetWidth > fig.clientWidth ? px - 12 - tip.offsetWidth : px + 12;
      tip.style.left = `${Math.max(4, left)}px`;
      tip.style.top = `${body.offsetTop + 8}px`;
    };
    const hide = () => {
      active = null;
      tip.hidden = true;
      cross.setAttribute("visibility", "hidden");
      hoverDots.forEach((d) => d.setAttribute("visibility", "hidden"));
    };
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const i = n === 1 ? 0 : Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1));
      show(Math.min(n - 1, Math.max(0, i)));
    });
    hit.addEventListener("pointerleave", hide);
    fig.onfocus = () => show(active ?? n - 1);
    fig.onblur = hide;
    fig.onkeydown = (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        show(Math.min(n - 1, Math.max(0, (active ?? n - 1) + (e.key === "ArrowRight" ? 1 : -1))));
      } else if (e.key === "Escape") hide();
    };
  }

  fig._render = render;
  render();
  let pending;
  new ResizeObserver(() => { clearTimeout(pending); pending = setTimeout(render, 80); }).observe(body);
}

document.querySelectorAll("figure.chart").forEach(lineChart);

// -- real-bet form: <select data-sides="fieldsetId"> names the team buttons after the chosen game's
//    teams, and clears the choice when the game changes so a side is never carried to another game --
document.querySelectorAll("select[data-sides]").forEach((sel) => {
  const box = document.getElementById(sel.dataset.sides);
  if (!box) return;
  sel.addEventListener("change", () => {
    const opt = sel.selectedOptions[0];
    box.querySelectorAll("[data-side]").forEach((span) => {
      const side = span.dataset.side;
      span.textContent = (opt && opt.dataset[side]) || `${side[0].toUpperCase()}${side.slice(1)} team`;
    });
    box.querySelectorAll('input[name="side"]').forEach((r) => { r.checked = false; });
  });
});
