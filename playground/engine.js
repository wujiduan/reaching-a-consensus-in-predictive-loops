/**
 * Predictive Loops numerical model.
 * Implements Algorithm 1 of Wu, Abebe & Mendler-Dünner (2026), arXiv:2603.12137v2.
 * Synthetic networks, synthetic metadata and ridge regularization are exploratory
 * additions, not a reproduction of the paper's Yelp/Pokec experiments.
 */

export const DEFAULTS = Object.freeze({
  n: 80, topology: 'communities', density: 0.10, mixing: 0.05,
  alpha: 0.5, beta: 0.85, heterogeneity: 'uniform', stubbornFraction: 0.2,
  k: 10, rounds: 120, distribution: 'bimodal', spread: 0.10,
  mean: 0.5, separation: 0.65, alignment: 'random', seed: 42,
  model: 'perfect', observed: 0.8, sampling: 'random',
  targetNode: 0, target: 0.9, targetBeta: -1, receiverNode: -1, ridge: 0.01,
});

export const PRESETS = Object.freeze([
  { id: 'consensus', title: 'The pull toward consensus', description: 'Strong platform influence draws a divided population together.', config: { ...DEFAULTS, topology: 'communities', density: 0.22, mixing: 0.025, alignment: 'community', alpha: 0.55, beta: 0.97, k: 20, rounds: 160 } },
  { id: 'stubborn', title: 'Stubborn, but not immune', description: 'Compare a platform-free network with a predictive loop.', config: { ...DEFAULTS, topology: 'ring', alpha: 0.25, beta: 0.95, k: 0, rounds: 160 } },
  { id: 'spillover', title: 'Influence beyond the edges', description: 'Unobserved people ignore peers, yet learn through mean predictions.', config: { ...DEFAULTS, topology: 'communities', density: 0.25, mixing: 0.025, model: 'mean', observed: 0.5, sampling: 'community', heterogeneity: 'stubborn', stubbornFraction: 0.5, alignment: 'community', alpha: 0.6, beta: 0.97, k: 20, rounds: 200 } },
  { id: 'intervention', title: 'One targeted prediction', description: 'Pin one prediction and trace its influence through the network.', config: { ...DEFAULTS, topology: 'scaleFree', model: 'intervention', targetNode: 0, target: 0.95, alpha: 0.6, beta: 0.95, k: 20, rounds: 160 } },
  { id: 'disconnected', title: 'Two separate worlds', description: 'Disconnected communities need not agree under perfect prediction.', config: { ...DEFAULTS, topology: 'disconnected', density: 0.2, alignment: 'community', alpha: 0.6, beta: 0.99, k: 20, rounds: 200 } },
]);

const ENUMS = {
  topology: ['communities', 'random', 'smallworld', 'scaleFree', 'ring', 'complete', 'disconnected'],
  heterogeneity: ['uniform', 'random', 'stubborn'],
  distribution: ['bimodal', 'uniform', 'normal', 'skewed', 'custom'],
  alignment: ['random', 'community'], model: ['perfect', 'mean', 'linear', 'intervention'],
  sampling: ['random', 'community'],
};
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const unit = x => clamp(x, 0, 1);
const numeric = (x, fallback) => Number.isFinite(Number(x)) ? Number(x) : fallback;

export function normalizeConfig(input = {}) {
  const c = { ...DEFAULTS };
  for (const key of Object.keys(ENUMS)) c[key] = ENUMS[key].includes(input[key]) ? input[key] : DEFAULTS[key];
  const limits = {
    n: [20, 200], density: [0, 1], mixing: [0, 1], alpha: [0, 0.99], beta: [0, 1],
    stubbornFraction: [0, 1], k: [0, 100], rounds: [20, 500], spread: [0, 0.5],
    mean: [0, 1], separation: [0, 1], observed: [0.01, 1], target: [0, 1], targetBeta: [-1, 1], ridge: [0, 10],
    seed: [0, 4294967295],
  };
  for (const [key, [lo, hi]] of Object.entries(limits)) c[key] = clamp(numeric(input[key], DEFAULTS[key]), lo, hi);
  for (const key of ['n', 'k', 'rounds', 'seed']) c[key] = Math.round(c[key]);
  c.targetNode = Math.round(clamp(numeric(input.targetNode, DEFAULTS.targetNode), 0, c.n - 1));
  c.receiverNode = Math.round(clamp(numeric(input.receiverNode, -1), -1, c.n - 1));
  if (input.customOpinions !== undefined) {
    if (!Array.isArray(input.customOpinions) || input.customOpinions.length !== c.n) throw new Error(`Custom opinions must contain exactly ${c.n} numbers.`);
    if (input.customOpinions.some(x => typeof x !== 'number' || !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('Custom opinions must be finite numbers between 0 and 1.');
    c.customOpinions = [...input.customOpinions];
  }
  if (c.distribution === 'custom' && !c.customOpinions) throw new Error(`Add ${c.n} custom opinions between 0 and 1 before running this distribution.`);
  return c;
}

/** Independent deterministic streams keep data fixed when unrelated controls change. */
export function seededRandom(seed, stream = '') {
  let h = (2166136261 ^ (seed >>> 0)) >>> 0;
  for (let i = 0; i < stream.length; i++) h = Math.imul(h ^ stream.charCodeAt(i), 16777619) >>> 0;
  return () => {
    h = (h + 0x6D2B79F5) >>> 0;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  return Math.sqrt(-2 * Math.log(Math.max(1e-15, rng()))) * Math.cos(2 * Math.PI * rng());
}

function gamma(shape, rng) {
  if (shape < 1) return gamma(shape + 1, rng) * Math.pow(Math.max(1e-15, rng()), 1 / shape);
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = gaussian(rng), v0 = 1 + c * x;
    if (v0 <= 0) continue;
    const v = v0 ** 3, u = rng();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(Math.max(1e-15, u)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function shuffle(values, rng) {
  const a = [...values];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

function connectedComponents(neighbors) {
  const seen = new Uint8Array(neighbors.length), components = [];
  for (let start = 0; start < neighbors.length; start++) {
    if (seen[start]) continue;
    const comp = [], queue = [start]; seen[start] = 1;
    for (let head = 0; head < queue.length; head++) {
      const i = queue[head]; comp.push(i);
      for (const j of neighbors[i]) if (!seen[j]) { seen[j] = 1; queue.push(j); }
    }
    components.push(comp);
  }
  return components;
}

function buildGraph(c) {
  const rng = seededRandom(c.seed, 'graph'), n = c.n;
  const sets = Array.from({ length: n }, () => new Set());
  const groups = Array.from({ length: n }, (_, i) => i < Math.ceil(n / 2) ? 0 : 1);
  const add = (i, j) => { if (i !== j) { sets[i].add(j); sets[j].add(i); } };
  const remove = (i, j) => { sets[i].delete(j); sets[j].delete(i); };
  let attachmentCount = null, latticeDegree = null;
  if (c.topology === 'complete') {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) add(i, j);
  } else if (c.topology === 'ring') {
    for (let i = 0; i < n; i++) add(i, (i + 1) % n);
  } else if (c.topology === 'smallworld') {
    const half = Math.min(Math.floor((n - 1) / 2), Math.max(1, Math.round(c.density * (n - 1) / 2)));
    latticeDegree = half * 2;
    const original = [];
    for (let i = 0; i < n; i++) for (let k = 1; k <= half; k++) { add(i, (i + k) % n); original.push([i, (i + k) % n]); }
    for (const [i, j] of original) {
      if (rng() >= c.mixing) continue;
      const candidates = [];
      for (let v = 0; v < n; v++) if (v !== i && !sets[i].has(v)) candidates.push(v);
      if (candidates.length) { remove(i, j); add(i, candidates[Math.floor(rng() * candidates.length)]); }
    }
  } else if (c.topology === 'scaleFree') {
    const m = Math.max(1, Math.min(12, Math.round(c.density * (n - 1) / 2)));
    attachmentCount = m;
    for (let i = 0; i < m + 1; i++) for (let j = i + 1; j < m + 1; j++) add(i, j);
    for (let i = m + 1; i < n; i++) {
      for (let e = 0; e < m; e++) {
        let total = 0;
        for (let j = 0; j < i; j++) if (!sets[i].has(j)) total += sets[j].size + 1;
        let draw = rng() * total, selected = -1;
        for (let j = 0; j < i; j++) {
          if (sets[i].has(j)) continue;
          draw -= sets[j].size + 1;
          if (draw <= 0) { selected = j; break; }
        }
        if (selected >= 0) add(i, selected);
      }
    }
  } else {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const u = rng();
      const same = groups[i] === groups[j];
      const p = c.topology === 'random' ? c.density : same ? c.density : c.topology === 'disconnected' ? 0 : c.mixing;
      if (u < p) add(i, j);
    }
  }
  // Connected synthetic scenarios are conditioned by joining sampled components.
  // The explicit disconnected scenario is conditioned only within each group.
  let addedEdges = 0;
  if (c.topology === 'disconnected') {
    for (let group = 0; group < 2; group++) {
      const components = connectedComponents(sets).filter(comp => groups[comp[0]] === group);
      for (let k = 1; k < components.length; k++) { add(components[k - 1][0], components[k][0]); addedEdges++; }
    }
  } else {
    const components = connectedComponents(sets);
    for (let k = 1; k < components.length; k++) { add(components[k - 1][0], components[k][0]); addedEdges++; }
  }
  const neighbors = sets.map(set => [...set].sort((a, b) => a - b));
  const edges = [];
  for (let i = 0; i < n; i++) for (const j of neighbors[i]) if (i < j) edges.push([i, j]);
  const components = connectedComponents(neighbors);
  return { n, edges, neighbors, degree: neighbors.map(row => row.length), groups, components, componentCount: components.length, addedEdges, attachmentCount, latticeDegree };
}

function susceptibilities(c, n) {
  const rng = seededRandom(c.seed, 'susceptibilities');
  const alpha = [], beta = [];
  const stubbornCount = Math.round(n * c.stubbornFraction);
  for (let i = 0; i < n; i++) {
    // Draw both noises independently of selected mode to preserve parameter comparisons.
    const aNoise = gaussian(rng), bNoise = gaussian(rng);
    alpha.push(c.heterogeneity === 'random' ? clamp(c.alpha + 0.1 * aNoise, 0, 0.99) : c.heterogeneity === 'stubborn' && i >= n - stubbornCount ? 0 : c.alpha);
    beta.push(c.heterogeneity === 'random' ? unit(c.beta + 0.1 * bNoise) : c.beta);
  }
  if (c.model === 'intervention') {
    if (c.targetBeta >= 0) beta[c.targetNode] = c.targetBeta;
    if (c.receiverNode >= 0 && c.receiverNode !== c.targetNode) beta[c.receiverNode] = 0;
  }
  return { alpha, beta };
}

function observationMask(c, groups) {
  const ids = Array.from({ length: groups.length }, (_, i) => i);
  const order = c.sampling === 'community' ? ids.sort((a, b) => groups[a] - groups[b] || a - b) : shuffle(ids, seededRandom(c.seed, 'observations'));
  const count = Math.max(1, Math.round(groups.length * c.observed));
  const mask = Array(groups.length).fill(false);
  for (let k = 0; k < count; k++) mask[order[k]] = true;
  return mask;
}

export function createWorld(input = {}) {
  const c = normalizeConfig(input), graph = buildGraph(c);
  const rng = seededRandom(c.seed, 'innate-opinions');
  const modes = c.alignment === 'community' ? graph.groups : shuffle(graph.groups, seededRandom(c.seed, 'opinion-alignment'));
  const xstar = Array.from({ length: c.n }, (_, i) => {
    if (c.distribution === 'custom') return c.customOpinions[i];
    if (c.distribution === 'uniform') return rng();
    if (c.distribution === 'normal') return unit(c.mean + c.spread * gaussian(rng));
    if (c.distribution === 'skewed') { const a = gamma(2, rng), b = gamma(5, rng); return a / (a + b); }
    return unit(c.mean + (modes[i] === 0 ? -1 : 1) * c.separation / 2 + c.spread * gaussian(rng));
  });
  const featureRng = seededRandom(c.seed, 'metadata');
  const features = xstar.map((x, i) => [1, graph.groups[i] * 2 - 1, 2 * unit(x + 0.2 * gaussian(featureRng)) - 1]);
  const world = {
    ...graph, ...susceptibilities(c, c.n), xstar, features,
    observed: observationMask(c, graph.groups), config: c,
  };
  world.positions = computeLayout(world);
  return world;
}

/** Layout coordinates are only visual; they never enter the opinion dynamics. */
export function computeLayout(world) {
  const { n, edges, groups } = world;
  const c = world.config || DEFAULTS;
  const rng = seededRandom(c.seed, 'layout');
  if (c.topology === 'ring') return Array.from({ length: n }, (_, i) => ({ x: 0.5 + 0.42 * Math.cos(i * 2 * Math.PI / n - Math.PI / 2), y: 0.5 + 0.42 * Math.sin(i * 2 * Math.PI / n - Math.PI / 2) }));
  if (c.topology === 'complete') return Array.from({ length: n }, (_, i) => { const a = i * Math.PI * (3 - Math.sqrt(5)), r = 0.42 * Math.sqrt((i + 0.5) / n); return { x: 0.5 + r * Math.cos(a), y: 0.5 + r * Math.sin(a) }; });
  const clustered = c.topology === 'communities' || c.topology === 'disconnected';
  const groupCounts = [groups.filter(g => g === 0).length, groups.filter(g => g === 1).length];
  const groupIndices = [0, 0];
  const points = Array.from({ length: n }, (_, i) => {
    const index = clustered ? groupIndices[groups[i]]++ : i;
    const count = clustered ? groupCounts[groups[i]] : n;
    const a = index * Math.PI * (3 - Math.sqrt(5)) + (rng() - 0.5) * 0.12;
    const r = Math.sqrt((index + 0.5) / count) * (clustered ? 0.19 : 0.32);
    return { x: (clustered ? groups[i] ? 0.76 : 0.24 : 0.5) + r * Math.cos(a), y: 0.5 + r * Math.sin(a) };
  });
  for (let iter = 0; iter < 100; iter++) {
    const dx = new Float64Array(n), dy = new Float64Array(n);
    const repulsion = 0.00055 / Math.sqrt(n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const xx = points[i].x - points[j].x, yy = points[i].y - points[j].y;
      const betweenGroups = clustered && groups[i] !== groups[j];
      const inv = repulsion * (betweenGroups ? 0.08 : 1) / (xx * xx + yy * yy + 0.0002);
      dx[i] += xx * inv; dy[i] += yy * inv; dx[j] -= xx * inv; dy[j] -= yy * inv;
    }
    for (const [i, j] of edges) {
      const spring = clustered && groups[i] !== groups[j] ? 0.001 : 0.005;
      const xx = (points[j].x - points[i].x) * spring, yy = (points[j].y - points[i].y) * spring;
      dx[i] += xx; dy[i] += yy; dx[j] -= xx; dy[j] -= yy;
    }
    const cooling = 0.8 * (1 - iter / 150);
    for (let i = 0; i < n; i++) {
      const cx = clustered ? groups[i] ? 0.76 : 0.24 : 0.5;
      dx[i] += (cx - points[i].x) * 0.035;
      dy[i] += (0.5 - points[i].y) * 0.035;
      const magnitude = Math.hypot(dx[i], dy[i]);
      const step = cooling * Math.min(1, 0.025 / Math.max(magnitude, 1e-12));
      // Bound motion, not positions: hard viewport clamps flatten dense clouds.
      points[i].x += dx[i] * step;
      points[i].y += dy[i] * step;
    }
  }
  const minX = Math.min(...points.map(p => p.x)), maxX = Math.max(...points.map(p => p.x));
  const minY = Math.min(...points.map(p => p.y)), maxY = Math.max(...points.map(p => p.y));
  return points.map(p => ({ x: 0.07 + 0.86 * (p.x - minX) / Math.max(0.001, maxX - minX), y: 0.07 + 0.86 * (p.y - minY) / Math.max(0.001, maxY - minY) }));
}

/** LU factorization with partial pivoting. The supplied matrix is overwritten. */
function factorMatrix(lu) {
  const n = lu.length;
  const permutation = Array.from({ length: n }, (_, i) => i);
  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(lu[i][k]) > Math.abs(lu[pivot][k])) pivot = i;
    if (Math.abs(lu[pivot][k]) < 1e-14) throw new Error('The linear system is numerically singular.');
    if (pivot !== k) { [lu[k], lu[pivot]] = [lu[pivot], lu[k]]; [permutation[k], permutation[pivot]] = [permutation[pivot], permutation[k]]; }
    for (let i = k + 1; i < n; i++) {
      if (lu[i][k] === 0) continue;
      lu[i][k] /= lu[k][k];
      const multiplier = lu[i][k];
      for (let j = k + 1; j < n; j++) lu[i][j] -= multiplier * lu[k][j];
    }
  }
  return { lu, permutation };
}
function solveFactored({ lu, permutation }, rhs) {
  const n = lu.length, result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let value = rhs[permutation[i]];
    for (let j = 0; j < i; j++) value -= lu[i][j] * result[j];
    result[i] = value;
  }
  for (let i = n - 1; i >= 0; i--) {
    let value = result[i];
    for (let j = i + 1; j < n; j++) value -= lu[i][j] * result[j];
    result[i] = value / lu[i][i];
  }
  return result;
}

function peerMatrix(world) {
  const { n, alpha, neighbors } = world;
  return Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    if (!neighbors[i].length) row[i] = alpha[i];
    else for (const j of neighbors[i]) row[j] = alpha[i] / neighbors[i].length;
    return row;
  });
}

const fjFactors = new WeakMap();
function getFJFactor(world) {
  let factor = fjFactors.get(world);
  if (!factor) {
    const system = peerMatrix(world);
    for (let i = 0; i < world.n; i++) {
      for (let j = 0; j < world.n; j++) system[i][j] = -system[i][j];
      system[i][i] += 1;
    }
    factor = factorMatrix(system);
    fjFactors.set(world, factor);
  }
  return factor;
}

const finiteKernels = new WeakMap();
function multiplyMatrices(a, b) {
  const n = a.length, result = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const row = result[i];
    for (let k = 0; k < n; k++) {
      const coefficient = a[i][k];
      if (coefficient === 0) continue;
      const brow = b[k];
      for (let j = 0; j < n; j++) row[j] += coefficient * brow[j];
    }
  }
  return result;
}

function sumMatrices(a, b) {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]));
}

function prepareDensePeerResponse(world, k, rounds) {
  if (k <= 1) return;
  const n = world.n, nonzero = world.neighbors.reduce((sum, row) => sum + Math.max(1, row.length), 0);
  const multiplications = Math.floor(Math.log2(k)) + k.toString(2).replace(/0/g, '').length - 1;
  const directWork = nonzero * k * rounds;
  const preparedWork = 2 * n ** 3 * multiplications + n ** 2 * rounds;
  if (directWork >= 2 * preparedWork && directWork >= 1e7) preparePeerKernel(world, k);
}

function preparePeerKernel(world, k) {
  const cached = finiteKernels.get(world);
  if (cached?.k === k) return cached.kernel;
  const n = world.n;
  // For each block, keep A^k and I + A + ... + A^(k-1). Concatenation
  // uses only nonnegative arithmetic and preserves exact structural zeros.
  let blockPower = peerMatrix(world);
  let blockSum = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    row[i] = 1;
    return row;
  });
  let power = null, sum = null, exponent = k;
  while (exponent) {
    if (exponent & 1) {
      sum = power ? sumMatrices(sum, multiplyMatrices(power, blockSum)) : blockSum;
      power = power ? multiplyMatrices(power, blockPower) : blockPower;
    }
    exponent = Math.floor(exponent / 2);
    if (exponent) {
      blockSum = sumMatrices(blockSum, multiplyMatrices(blockPower, blockSum));
      blockPower = multiplyMatrices(blockPower, blockPower);
    }
  }
  const kernel = sum.map((row, i) => row.map((value, j) => value * (1 - world.alpha[j]) + power[i][j]));
  finiteKernels.set(world, { k, kernel });
  return kernel;
}

function solveFJ(z, world) {
  const rhs = z.map((value, i) => (1 - world.alpha[i]) * value);
  return Array.from(solveFactored(getFJFactor(world), rhs), unit);
}

/** k=0 denotes the exact equilibrium, not zero peer interactions. */
export function peerResponse(z, world, k) {
  if (k === 0) return solveFJ(z, world);
  const prepared = finiteKernels.get(world);
  if (prepared?.k === k) {
    return prepared.kernel.map(row => {
      let value = 0;
      for (let j = 0; j < z.length; j++) value += row[j] * z[j];
      return unit(value);
    });
  }
  const { n, alpha, neighbors } = world;
  const anchor = z.map((value, i) => (1 - alpha[i]) * value);
  let x = [...z], next = new Array(n);
  for (let step = 0; step < k; step++) {
    for (let i = 0; i < n; i++) {
      let average = 0;
      if (neighbors[i].length) { for (const j of neighbors[i]) average += x[j]; average /= neighbors[i].length; }
      else average = x[i]; // Defined isolated-node extension: self-influence.
      next[i] = anchor[i] + alpha[i] * average;
    }
    [x, next] = [next, x];
  }
  return x;
}

/** Eigen pseudoinverse for a symmetric 2x2 matrix (unpenalized intercept). */
function solveSymmetric2(a, b, d, u, v) {
  const angle = 0.5 * Math.atan2(2 * b, a - d);
  const c = Math.cos(angle), s = Math.sin(angle);
  const l1 = a * c * c + 2 * b * c * s + d * s * s;
  const l2 = a * s * s - 2 * b * c * s + d * c * c;
  const threshold = Math.max(Math.abs(l1), Math.abs(l2), 1) * 1e-12;
  const q1 = Math.abs(l1) > threshold ? (c * u + s * v) / l1 : 0;
  const q2 = Math.abs(l2) > threshold ? (-s * u + c * v) / l2 : 0;
  return [c * q1 - s * q2, s * q1 + c * q2];
}

export function applyPredictor(x, world, config = world.config || DEFAULTS) {
  const c = config, n = world.n;
  if (c.model === 'perfect') return [...x];
  if (c.model === 'intervention') { const f = [...x]; f[clamp(c.targetNode, 0, n - 1)] = c.target; return f; }
  const observedIds = [];
  for (let i = 0; i < n; i++) if (world.observed[i]) observedIds.push(i);
  if (!observedIds.length) throw new Error('At least one observed individual is required.');
  const count = observedIds.length;
  const meanY = observedIds.reduce((sum, i) => sum + x[i], 0) / count;
  if (c.model === 'mean') return x.map((value, i) => world.observed[i] ? value : meanY);
  if (c.model === 'linear') {
    let meanA = 0, meanB = 0;
    for (const i of observedIds) { meanA += world.features[i][1] / count; meanB += world.features[i][2] / count; }
    let aa = count * c.ridge, ab = 0, bb = count * c.ridge, ay = 0, by = 0;
    for (const i of observedIds) {
      const a = world.features[i][1] - meanA, b = world.features[i][2] - meanB, y = x[i] - meanY;
      aa += a * a; ab += a * b; bb += b * b; ay += a * y; by += b * y;
    }
    const [slopeA, slopeB] = solveSymmetric2(aa, ab, bb, ay, by);
    return world.features.map(row => unit(meanY + slopeA * (row[1] - meanA) + slopeB * (row[2] - meanB)));
  }
  throw new Error(`Unknown prediction model: ${c.model}`);
}

export function metrics(x, world) {
  const n = x.length, origin = x[0];
  const mean = origin + x.reduce((sum, value) => sum + (value - origin), 0) / n;
  const variance = x.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n;
  let disagreement = 0;
  for (const [i, j] of world.edges) disagreement += (x[i] - x[j]) ** 2;
  disagreement = world.edges.length ? disagreement / world.edges.length : 0;
  const min = Math.min(...x), max = Math.max(...x);
  return { mean, variance, disagreement, range: max - min, min, max, sd: Math.sqrt(variance) };
}

function reuseWorld(c, source) {
  // Reuse geometry/data for parameter sweeps, but update model-dependent inputs.
  // A handcrafted test world with no config is respected verbatim.
  if (!source.config) return source;
  const n = source.n;
  return { ...source, ...susceptibilities(c, n), observed: observationMask(c, source.groups), config: { ...c, n } };
}

// Assemble the outer update x_next = B x + offset for the three affine policies.
function affineSystem(world, config) {
  const { n, beta, xstar } = world;
  const observed = xstar.map((_, i) => i).filter(i => world.observed[i]);
  const target = config.model === 'intervention' ? config.targetNode : -1;
  const B = Array.from({ length: n }, () => new Float64Array(n));
  const offset = new Float64Array(n), anchored = new Uint8Array(n);
  const kernel = config.k > 0 ? preparePeerKernel(world, config.k) : null;
  const basis = Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    basis[j] = 1;
    const column = kernel ? null : peerResponse(basis, world, config.k);
    basis[j] = 0;
    for (let i = 0; i < n; i++) {
      const influence = kernel ? kernel[i][j] : column[i];
      if (influence === 0) continue;
      offset[i] += influence * (1 - beta[j]) * xstar[j];
      if (beta[j] < 1 || j === target) anchored[i] = 1;
      const weight = influence * beta[j];
      if (j === target) offset[i] += weight * config.target;
      else if (config.model === 'mean' && !world.observed[j]) {
        for (const source of observed) B[i][source] += weight / observed.length;
      } else B[i][j] += weight;
    }
  }
  return { B, offset, anchored };
}

function reachable(edges, starts) {
  const visited = new Set(starts), queue = [...starts];
  for (let k = 0; k < queue.length; k++) {
    for (const j of edges[queue[k]]) if (!visited.has(j)) {
      visited.add(j);
      queue.push(j);
    }
  }
  return visited;
}

function closedClasses(B, anchored) {
  const outgoing = B.map(row => Array.from(row.keys()).filter(j => row[j] > 0));
  const incoming = B.map((_, i) => []);
  outgoing.forEach((row, i) => row.forEach(j => incoming[j].push(i)));
  const leadsToAnchor = reachable(incoming, Array.from(anchored.keys()).filter(i => anchored[i]));
  const classified = new Set(leadsToAnchor), classes = [];
  for (let i = 0; i < B.length; i++) {
    if (classified.has(i)) continue;
    const forward = reachable(outgoing, [i]);
    const backward = reachable(incoming, [i]);
    const component = [...forward].filter(j => backward.has(j));
    component.forEach(j => classified.add(j));
    if (component.every(j => outgoing[j].every(k => backward.has(k)))) classes.push(component);
  }
  return classes;
}

function stationaryAverage(B, ids, xstar) {
  if (ids.length === 1) return xstar[ids[0]];
  // Solve the stationary balance equations with one row replaced by normalization.
  const matrix = ids.map((i, row) => Float64Array.from(ids, (j, col) => {
    if (row === 0) return 1;
    if (col === row) return ids.reduce((mass, k) => mass + (i === k ? 0 : B[i][k]), 0);
    return -B[j][i];
  }));
  const rhs = new Float64Array(ids.length);
  rhs[0] = 1;
  const weights = solveFactored(factorMatrix(matrix), rhs);
  return unit(ids.reduce((sum, i, j) => sum + weights[j] * xstar[i], 0));
}

function nextOpinions(x, world, config) {
  const f = applyPredictor(x, world, config);
  const z = world.xstar.map((value, i) => (1 - world.beta[i]) * value + world.beta[i] * f[i]);
  return peerResponse(z, world, config.k);
}

function equilibriumFor(world, config, sameKBaseline) {
  if (world.beta.every(b => b === 0)) return { x: [...sameKBaseline], closedClassCount: 0 };
  if (config.model === 'linear') return null;
  const { B, offset, anchored } = affineSystem(world, config);
  const x = new Array(world.n).fill(null);
  const classes = world.beta.some(b => b === 1) ? closedClasses(B, anchored) : [];
  for (const ids of classes) {
    const value = stationaryAverage(B, ids, world.xstar);
    for (const i of ids) x[i] = value;
  }
  const unknown = world.xstar.map((_, i) => i).filter(i => x[i] === null);
  if (unknown.length) {
    const matrix = unknown.map(i => Float64Array.from(unknown, j => (i === j ? 1 : 0) - B[i][j]));
    const rhs = unknown.map(i => B[i].reduce((sum, weight, j) => sum + (x[j] === null ? 0 : weight * x[j]), offset[i]));
    const solved = solveFactored(factorMatrix(matrix), rhs);
    unknown.forEach((i, k) => { x[i] = unit(solved[k]); });
  }
  return { x, closedClassCount: classes.length };
}

export function simulate(input = {}, optionalWorld) {
  const config = normalizeConfig(input);
  const world = optionalWorld ? reuseWorld(config, optionalWorld) : createWorld(config);
  config.n = world.n;
  config.targetNode = clamp(config.targetNode, 0, world.n - 1);
  const { n, xstar, alpha, beta } = world;
  const warnings = [];
  if (world.addedEdges) warnings.push(`${world.addedEdges} linking edge${world.addedEdges === 1 ? '' : 's'} added to connect sampled ${config.topology === 'disconnected' ? 'within-community ' : ''}components.`);
  if (world.componentCount > 1) warnings.push('Disconnected-network extension: global consensus is not guaranteed.');
  if (config.heterogeneity === 'stubborn' && alpha.some(a => a === 0)) warnings.push('Some people ignore peer influence (αᵢ = 0); the full-observation consensus theorem requires αᵢ > 0 for every person.');
  if (config.beta === 1 || beta.some(b => b === 1)) warnings.push('At βᵢ = 1 the innate anchor vanishes for those people; fixed-point uniqueness and convergence require care.');
  if (config.model === 'linear') warnings.push('Linear model uses synthetic fixed metadata; ridge regularization is an exploratory addition.');
  if (config.model === 'perfect' || config.model === 'intervention') warnings.push('Perfect-copy and targeted-intervention models observe everyone; the observation fraction applies only to mean and linear models.');
  if (config.distribution === 'normal' || config.distribution === 'bimodal') warnings.push('Normal samples are clipped to the opinion interval [0, 1].');
  const baseline = peerResponse(xstar, world, 0);
  const sameKBaseline = config.k === 0 ? [...baseline] : peerResponse(xstar, world, config.k);
  prepareDensePeerResponse(world, config.k, config.rounds);
  let f = applyPredictor(xstar, world, config), previous = [...xstar];
  const initialMSE = f.reduce((sum, value, i) => sum + (value - xstar[i]) ** 2, 0) / n;
  const history = [{ round: 0, x: [...xstar], f: [...f], ...metrics(xstar, world), delta: 0, predictionError: initialMSE }];
  let convergenceRound = null;
  for (let round = 1; round <= config.rounds; round++) {
    const z = xstar.map((value, i) => (1 - beta[i]) * value + beta[i] * f[i]);
    const x = peerResponse(z, world, config.k);
    let delta = 0, predictionError = 0;
    for (let i = 0; i < n; i++) { delta = Math.max(delta, Math.abs(x[i] - previous[i])); predictionError += (f[i] - x[i]) ** 2; }
    predictionError /= n;
    history.push({ round, x: [...x], f: [...f], ...metrics(x, world), delta, predictionError });
    if (convergenceRound === null && round >= 2 && delta < 1e-7) convergenceRound = round;
    previous = x;
    f = applyPredictor(x, world, config);
    if (delta === 0) {
      // A deterministic update with an identical state is exactly stationary.
      // Preserve the full requested timeline without recomputing the fixed point.
      if (convergenceRound === null) convergenceRound = Math.max(2, round);
      const fixed = history[history.length - 1];
      for (let future = round + 1; future <= config.rounds; future++) history.push({ ...fixed, round: future, x: [...x], f: [...f], delta: 0 });
      break;
    }
  }
  const final = history[history.length - 1];
  const nextX = nextOpinions(final.x, world, config);
  const residual = Math.max(...nextX.map((value, i) => Math.abs(value - final.x[i])));
  let equilibrium = null;
  let equilibriumIssue = null;
  try {
    const solved = equilibriumFor(world, config, sameKBaseline);
    if (solved) {
      const next = nextOpinions(solved.x, world, config);
      const equilibriumResidual = Math.max(...next.map((value, i) => Math.abs(value - solved.x[i])));
      if (equilibriumResidual > 1e-9) throw new Error('The fixed-point residual exceeds 10⁻⁹.');
      equilibrium = { ...solved, f: applyPredictor(solved.x, world, config), ...metrics(solved.x, world), residual: equilibriumResidual };
    }
  } catch (error) {
    equilibriumIssue = error.message;
  }
  return {
    config, world, history, equilibrium, equilibriumIssue, baseline, baselineMetrics: metrics(baseline, world),
    sameKBaseline, sameKBaselineMetrics: metrics(sameKBaseline, world),
    initialMetrics: metrics(xstar, world), finalPrediction: f, final,
    converged: residual < 1e-7, convergenceRound, residual,
    convergenceTolerance: 1e-7, nearConsensus: final.range < 0.01,
    consensusTolerance: 0.01, warnings,
  };
}
