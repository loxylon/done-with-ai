/** Pure numerical core. No DOM, network, learned weights, or hidden simulation. */
export const GOALS = [
  { name: 'Древесина', short: 'Д', color: '#cf8b38', icon: 'wood' },
  { name: 'Камень', short: 'К', color: '#6297aa', icon: 'stone' },
  { name: 'Кирка', short: 'И', color: '#a08cce', icon: 'tool' },
  { name: 'Похожая цель', short: 'Д′', color: '#b7a361', icon: 'wood' },
];
export const ACTIONS = ['Безопасный путь', 'Рискованный путь', 'Смертельный шаг'];
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const sum = xs => xs.reduce((a, b) => a + b, 0);
export const argmax = xs => xs.indexOf(Math.max(...xs));

export function labels(goals, duplicate = false) {
  const active = new Set(goals.filter(g => Number.isInteger(g) && g >= 0 && g < 3));
  if (duplicate && active.has(0)) active.add(3);
  const count = duplicate ? 4 : 3;
  return Array.from({ length: count }, (_, g) => active.has(g) ? 1 / active.size : 0);
}

export function trajectory(state) {
  const result = state.frames.slice(0, state.live).map((goals, i) => ({
    goals, name: `s${i + 1}`, terminated: false, truncated: false, fatal: false,
  }));
  if (state.end === 'death') {
    result.push({ goals: [], name: '∅', terminated: true, truncated: false, fatal: true });
  } else if (state.end === 'success') result.at(-1).terminated = true;
  else if (state.end === 'timeout') result.at(-1).truncated = true;
  if (state.reset && state.end !== 'window') {
    result.push({ goals: [1], name: 's′1', terminated: false, truncated: false, fatal: false, reset: true });
    result.push({ goals: [1], name: 's′2', terminated: false, truncated: false, fatal: false, reset: true });
  }
  return result;
}

export function targets(transitions, anchor, options) {
  const { gamma, term, cap, method = 'corrected', duplicate = false } = options;
  if (!(gamma > 0 && gamma < 1) || !(term >= 0 && term < 1) || !Number.isInteger(cap) || cap < 1)
    throw new RangeError('Invalid discounts or offset cap');
  if (!Number.isInteger(anchor) || anchor < 0 || anchor >= transitions.length)
    throw new RangeError('Anchor outside trajectory');
  const G = duplicate ? 4 : 3;
  const empty = Array(G).fill(0);
  const endIndex = transitions.findIndex((s, i) => i >= anchor && (s.terminated || s.truncated));
  const last = endIndex < 0 ? transitions.length - 1 : endIndex;
  const futures = [];
  for (let j = anchor; j <= last && j < anchor + cap; j++) {
    if (transitions[j].fatal) continue;
    const offset = j - anchor;
    futures.push({ index: j, k: offset + 1, raw: (1 - gamma) * gamma ** offset,
      labels: labels(transitions[j].goals, duplicate) });
  }
  const normalizer = sum(futures.map(f => f.raw));
  futures.forEach(f => { f.weight = f.raw / normalizer; });
  let y = empty.map((_, g) => sum(futures.map(f => f.weight * f.labels[g])));
  const terminated = endIndex >= 0 && transitions[endIndex].terminated;
  const delta = terminated ? endIndex - anchor + Number(!transitions[endIndex].fatal) : Infinity;
  const retained = method === 'corrected' && terminated ? 1 - term ** delta : 1;
  if (method === 'corrected') y = y.map(x => x * retained);
  const mass = sum(y);
  if (method === 'classic') {
    return { y: mass > 0 ? y.map(x => x / mass) : empty, absorbing: 0,
      valid: mass > 0, futures, delta, retained: 1, normalizer, missing: 1 - mass };
  }
  return { y, absorbing: clamp(1 - mass, 0, 1), valid: true, futures, delta,
    retained, normalizer, missing: 1 - mass };
}

export function softmax(logits) {
  const largest = Math.max(...logits);
  const e = logits.map(x => Math.exp(x - largest));
  const total = sum(e);
  return e.map(x => x / total);
}
export function classProbabilities(logits) { return softmax([...logits, 0]); }
export function crossEntropy(logits, target) {
  const z = [...logits, 0];
  const largest = Math.max(...z);
  const logZ = largest + Math.log(sum(z.map(x => Math.exp(x - largest))));
  return sum(target.map((y, i) => y * (logZ - z[i])));
}
export function criticGradient(logits, target) {
  return classProbabilities(logits).slice(0, -1).map((p, i) => p - target[i]);
}
export function rng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6D2B79F5) >>> 0;
    let t = Math.imul(value ^ value >>> 15, 1 | value);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
export function drawOffset(futures, random) {
  const u = random(); let cumulative = 0;
  for (let i = 0; i < futures.length; i++) {
    cumulative += futures[i].weight;
    if (u < cumulative) return i;
  }
  return Math.max(0, futures.length - 1);
}
export function conditionalVariance(futures, g) {
  const mean = sum(futures.map(f => f.weight * f.labels[g]));
  return Math.max(0, sum(futures.map(f => f.weight * (f.labels[g] - mean) ** 2)));
}

/** PPO policy surrogate + teacher CE. Value/entropy omitted intentionally in this lab. */
export function ppo({ theta, old, advantage, clip, beta, progress, teacher }) {
  const policy = softmax([theta, 0, 0]);
  const ratio = policy[0] / old;
  const bounded = clamp(ratio, 1 - clip, 1 + clip);
  const raw = ratio * advantage;
  const clipped = bounded * advantage;
  const surrogate = Math.min(raw, clipped);
  const clippedActive = clipped < raw && (ratio < 1 - clip || ratio > 1 + clip);
  const policyGradient = clippedActive ? 0 : -advantage * policy[0] * (1 - policy[0]) / old;
  const effectiveBeta = beta * (1 - progress / 100);
  const bc = -Math.log(policy[teacher]);
  const bcGradient = policy[0] - Number(teacher === 0);
  return { policy, ratio, bounded, raw, clipped, surrogate, clippedActive, effectiveBeta,
    bc, policyLoss: -surrogate, loss: -surrogate + effectiveBeta * bc,
    policyGradient, bcGradient, gradient: policyGradient + effectiveBeta * bcGradient };
}
