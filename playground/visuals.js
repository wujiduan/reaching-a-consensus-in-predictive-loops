/** Responsive canvas views. Plot values always come from the simulation. */
const INK = '#193b38';
const TEAL = '#2a8e89';
const MUTED = '#7b8983';
const GRID = 'rgba(79, 105, 95, .10)';
const FONT = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export function opinionColor(value) {
  const v = clamp(Number.isFinite(value) ? value : .5, 0, 1);
  const a = v < .5 ? [219, 122, 83] : [212, 204, 179];
  const b = v < .5 ? [212, 204, 179] : [32, 127, 123];
  const t = v < .5 ? v * 2 : (v - .5) * 2;
  return '#' + a.map((c, i) => Math.round(c + (b[i] - c) * t).toString(16).padStart(2, '0')).join('');
}

function prepare(canvas) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width, height = rect.height;
  if (width < 1 || height < 1) return null;
  const dpr = Math.min(globalThis.devicePixelRatio || 1, 3);
  const pxW = Math.round(width * dpr), pxH = Math.round(height * dpr);
  if (canvas.width !== pxW || canvas.height !== pxH) {
    canvas.width = pxW;
    canvas.height = pxH;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.font = `10px ${FONT}`;
  ctx.fillStyle = MUTED;
  return {ctx, width, height};
}

function nodePoint(pos, width, height) {
  const px = pos?.x ?? .5, py = pos?.y ?? .5;
  const padding = 10;
  return {x: padding + px * (width - padding * 2), y: padding + py * (height - padding * 2)};
}

export function drawNetwork(canvas, world, opinions, options = {}) {
  const surface = prepare(canvas);
  if (!surface || !world?.positions?.length) return surface;
  const {ctx, width, height} = surface;
  const points = world.positions.map(p => nodePoint(p, width, height));
  const n = points.length;
  const baseRadius = clamp(7.4 - Math.sqrt(n) * .21, 3.8, 6.4);
  const selected = Number.isInteger(options.selected) ? options.selected : -1;
  const showObserved = Boolean(options.showObserved);
  const target = options.target ?? -1;
  const selectedNeighbors = selected >= 0 ? new Set([selected]) : null;
  if (selectedNeighbors) for (const [i, j] of world.edges || []) {
    if (i === selected) selectedNeighbors.add(j);
    if (j === selected) selectedNeighbors.add(i);
  }

  // A single path keeps even dense networks visually light and inexpensive.
  ctx.beginPath();
  for (const [i, j] of world.edges || []) {
    if (!points[i] || !points[j]) continue;
    ctx.moveTo(points[i].x, points[i].y);
    ctx.lineTo(points[j].x, points[j].y);
  }
  const edgeDensity = (world.edges?.length || 0) / Math.max(n, 1);
  ctx.strokeStyle = `rgba(63, 91, 83, ${edgeDensity > 18 ? .042 : edgeDensity > 8 ? .068 : .13})`;
  ctx.lineWidth = .75;
  ctx.stroke();

  if (selected >= 0 && points[selected]) {
    ctx.beginPath();
    for (const [i, j] of world.edges || []) if (i === selected || j === selected) {
      ctx.moveTo(points[i].x, points[i].y);
      ctx.lineTo(points[j].x, points[j].y);
    }
    ctx.strokeStyle = 'rgba(37, 103, 94, .38)';
    ctx.lineWidth = 1.15;
    ctx.stroke();
  }

  for (let i = 0; i < n; i++) {
    const p = points[i];
    const r = baseRadius + (i === selected ? 1.4 : 0);
    const color = opinionColor(opinions?.[i]);
    const observed = world.observed?.[i] !== false;
    ctx.globalAlpha = selectedNeighbors && !selectedNeighbors.has(i) ? .48 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.shadowColor = 'rgba(35, 69, 60, .10)';
    ctx.shadowBlur = 5;
    ctx.shadowOffsetY = 1.5;
    ctx.fillStyle = showObserved && !observed ? '#f5f4ec' : color;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    ctx.lineWidth = showObserved && !observed ? 1.6 : 1.3;
    ctx.strokeStyle = showObserved && !observed ? color : 'rgba(255,255,255,.95)';
    ctx.stroke();
    if (i === selected || i === target) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
      ctx.strokeStyle = i === target ? '#a8572e' : INK;
      ctx.lineWidth = i === target ? 2 : 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  return {...surface, points};
}

export function hitTestNetwork(canvas, world, clientX, clientY) {
  if (!canvas || !world?.positions) return -1;
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left, y = clientY - rect.top;
  let nearest = -1, minimum = 13 * 13;
  for (let i = 0; i < world.positions.length; i++) {
    const p = nodePoint(world.positions[i], rect.width, rect.height);
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d < minimum) {minimum = d; nearest = i;}
  }
  return nearest;
}

function line(ctx, x1, y1, x2, y2, color = GRID, width = 1) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function histogram(values, count) {
  const bins = Array(count).fill(0);
  for (const value of values || []) if (Number.isFinite(value)) {
    bins[Math.min(count - 1, Math.floor(clamp(value, 0, 1) * count))]++;
  }
  return bins;
}

export function drawDistribution(canvas, values, initial, baseline, options = {}) {
  const surface = prepare(canvas);
  if (!surface) return surface;
  const {ctx, width, height} = surface;
  const p = {left: 31, right: 12, top: 18, bottom: 33};
  const plotW = Math.max(1, width - p.left - p.right), plotH = Math.max(1, height - p.top - p.bottom);
  const count = width < 300 ? 16 : 20;
  const current = histogram(values, count), start = histogram(initial, count);
  const base = histogram(options.showBaseline === false ? [] : baseline, count);
  const maxCount = Math.max(2, ...current, ...start, ...base);
  const top = Math.max(2, Math.ceil(maxCount / 2) * 2);
  const xp = i => p.left + i / count * plotW;
  const yp = v => p.top + plotH * (1 - v / top);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const value of [0, top / 2, top]) {
    const y = yp(value);
    line(ctx, p.left, y, width - p.right, y);
    ctx.fillStyle = MUTED;
    ctx.fillText(String(value), p.left - 7, y);
  }
  const binW = plotW / count;
  for (let i = 0; i < count; i++) {
    if (start[i]) {
      ctx.fillStyle = 'rgba(133, 146, 132, .035)';
      ctx.fillRect(xp(i) + 1, yp(start[i]), binW - 2, plotH + p.top - yp(start[i]));
      ctx.strokeStyle = 'rgba(117, 134, 119, .30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(xp(i) + 1, yp(start[i]), binW - 2, plotH + p.top - yp(start[i]));
    }
    if (current[i]) {
      ctx.fillStyle = opinionColor((i + .5) / count);
      ctx.globalAlpha = .86;
      ctx.fillRect(xp(i) + binW * .16, yp(current[i]), binW * .68, plotH + p.top - yp(current[i]));
      ctx.globalAlpha = 1;
    }
  }
  if (options.showBaseline !== false && baseline?.length) {
    ctx.beginPath();
    for (let i = 0; i < count; i++) {
      const x = xp(i), y = yp(base[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      ctx.lineTo(x + binW, y);
    }
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(71, 102, 89, .57)';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = MUTED;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const v of [0, .5, 1]) ctx.fillText(v.toFixed(1), p.left + plotW * v, height - p.bottom + 8);
  ctx.textAlign = 'left';
  ctx.font = `9px ${FONT}`;
  ctx.fillText('People', p.left, 2);
  return {...surface, bins: current};
}

function timeAxes(surface, totalRounds, yMax, yLabel, formatY) {
  const {ctx, width, height} = surface;
  const p = {left: width < 300 ? 38 : 43, right: 14, top: 23, bottom: 37};
  const plotW = Math.max(1, width - p.left - p.right), plotH = Math.max(1, height - p.top - p.bottom);
  const xp = t => p.left + t / Math.max(totalRounds, 1) * plotW;
  const yp = v => p.top + (1 - v / Math.max(yMax, 1e-12)) * plotH;
  ctx.font = `10px ${FONT}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of [0, yMax / 2, yMax]) {
    const y = yp(v);
    line(ctx, p.left, y, width - p.right, y);
    ctx.fillStyle = MUTED;
    ctx.fillText(formatY(v), p.left - 7, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const ticks = totalRounds >= 4 ? [0, Math.round(totalRounds / 2), totalRounds] : [0, totalRounds];
  for (const t of ticks) ctx.fillText(String(t), xp(t), height - p.bottom + 8);
  ctx.font = `9px ${FONT}`;
  ctx.fillText('Retraining round', p.left + plotW / 2, height - 10);
  ctx.textAlign = 'left';
  ctx.fillText(yLabel, p.left, 3);
  return {p, plotW, plotH, xp, yp};
}

function trace(ctx, history, xp, yp, getter, start, end) {
  ctx.beginPath();
  let drawing = false;
  for (let i = start; i <= end; i++) {
    const value = getter(history[i]);
    if (!Number.isFinite(value)) {drawing = false; continue;}
    const x = xp(history[i].round ?? i), y = yp(value);
    if (!drawing) {ctx.moveTo(x, y); drawing = true;} else ctx.lineTo(x, y);
  }
}

function withPlotClip(ctx, axes, fn) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(axes.p.left - 1, axes.p.top - 1, axes.plotW + 2, axes.plotH + 2);
  ctx.clip();
  fn();
  ctx.restore();
}

function baselineLine(ctx, axes, value) {
  if (!Number.isFinite(value)) return;
  ctx.setLineDash([4, 4]);
  line(ctx, axes.p.left, axes.yp(value), axes.p.left + axes.plotW, axes.yp(value), 'rgba(90, 112, 95, .58)', 1.25);
  ctx.setLineDash([]);
}

function roundMarker(ctx, axes, t, y, color = INK) {
  const x = axes.xp(t);
  line(ctx, x, axes.p.top, x, axes.p.top + axes.plotH, 'rgba(53, 87, 74, .20)', .85);
  if (Number.isFinite(y)) {
    ctx.beginPath();
    ctx.arc(x, axes.yp(y), 3.6, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
}

export function drawTrajectories(canvas, history, round, options = {}) {
  const surface = prepare(canvas);
  if (!surface || !history?.length) return surface;
  const {ctx} = surface;
  const last = history.length - 1;
  const at = clamp(Math.round(round || 0), 0, last);
  const total = history[last].round ?? last;
  const axes = timeAxes(surface, total, 1, 'Opinion', v => v.toFixed(1));
  const n = history[0].x?.length || 0;
  withPlotClip(ctx, axes, () => {
    // All trajectories are retained; color is anchored in each person's innate opinion.
    for (let person = 0; person < n; person++) {
      ctx.strokeStyle = opinionColor(history[0].x[person]);
      ctx.lineWidth = .9;
      ctx.globalAlpha = n > 120 ? .055 : .10;
      trace(ctx, history, axes.xp, axes.yp, h => h.x?.[person], at, last);
      ctx.stroke();
      ctx.globalAlpha = n > 120 ? .25 : .42;
      trace(ctx, history, axes.xp, axes.yp, h => h.x?.[person], 0, at);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (options.showBaseline !== false) baselineLine(ctx, axes, options.baselineMean);
    ctx.strokeStyle = 'rgba(25,59,56,.17)';
    ctx.lineWidth = 1.6;
    trace(ctx, history, axes.xp, axes.yp, h => h.mean, at, last);
    ctx.stroke();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    trace(ctx, history, axes.xp, axes.yp, h => h.mean, 0, at);
    ctx.stroke();
    if (options.selected >= 0) {
      ctx.strokeStyle = '#a8572e';
      ctx.lineWidth = 1.9;
      trace(ctx, history, axes.xp, axes.yp, h => h.x[options.selected], 0, last);
      ctx.stroke();
    }
    roundMarker(ctx, axes, history[at].round ?? at, history[at].mean);
  });
  return {...surface, ...axes};
}

function niceCeiling(value) {
  if (!(value > 1e-10)) return .01;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const scaled = value / magnitude;
  return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 2.5 ? 2.5 : scaled <= 5 ? 5 : 10) * magnitude;
}

function varianceLabel(value) {
  if (value === 0) return '0';
  if (value < .0005) return value.toExponential(0);
  return value.toFixed(value >= .1 ? 2 : 3).replace(/0+$/, '').replace(/\.$/, '');
}

export function drawDisagreement(canvas, history, round, options = {}) {
  const surface = prepare(canvas);
  if (!surface || !history?.length) return surface;
  const {ctx} = surface;
  const last = history.length - 1;
  const at = clamp(Math.round(round || 0), 0, last);
  const total = history[last].round ?? last;
  const valueAt = h => h[options.measure || 'variance'];
  const observedMax = Math.max(0, ...history.map(valueAt), options.showBaseline === false ? 0 : options.baselineValue || 0);
  const yMax = niceCeiling(observedMax * 1.07);
  const axes = timeAxes(surface, total, yMax, options.measure === 'disagreement' ? 'Mean squared edge gap' : 'Opinion variance', varianceLabel);
  withPlotClip(ctx, axes, () => {
    const area = (end, opacity) => {
      trace(ctx, history, axes.xp, axes.yp, valueAt, 0, end);
      ctx.lineTo(axes.xp(history[end].round ?? end), axes.yp(0));
      ctx.lineTo(axes.xp(0), axes.yp(0));
      ctx.closePath();
      ctx.fillStyle = `rgba(42,142,137,${opacity})`;
      ctx.fill();
    };
    area(last, .035);
    area(at, .09);
    if (options.showBaseline !== false) baselineLine(ctx, axes, options.baselineValue);
    trace(ctx, history, axes.xp, axes.yp, valueAt, at, last);
    ctx.strokeStyle = 'rgba(42,142,137,.22)';
    ctx.lineWidth = 2;
    ctx.stroke();
    trace(ctx, history, axes.xp, axes.yp, valueAt, 0, at);
    ctx.strokeStyle = TEAL;
    ctx.lineWidth = 2.2;
    ctx.stroke();
    roundMarker(ctx, axes, history[at].round ?? at, valueAt(history[at]), TEAL);
  });
  return {...surface, ...axes};
}
