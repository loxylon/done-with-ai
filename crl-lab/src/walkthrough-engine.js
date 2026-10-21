/**
 * Исполняемая учебная модель Dual-CRL-PPO. Все градиенты аналитические.
 * В отличие от notebook, здесь табличные параметры и SGD вместо MLP и Adam:
 * каждое слагаемое и каждое изменение параметра можно показать без пропусков.
 */
import { rng, sum, softmax, clamp } from './math.js';

export const WIDTH = 4;
export const HEIGHT = 3;
export const START = 4;
export const WALL = 5;
export const DANGER = 7;
export const ACTIONS = [
  { name: 'Вверх', dx: 0, dy: -1 },
  { name: 'Вправо', dx: 1, dy: 0 },
  { name: 'Вниз', dx: 0, dy: 1 },
  { name: 'Влево', dx: -1, dy: 0 },
];
export const GOALS = [
  { id: 'A', name: 'Флажок', color: '#bd623e', test: s => s === 2 },
  { id: 'B', name: 'Кристалл', color: '#7c69ac', test: s => s === 11 },
  { id: 'C', name: 'Источник', color: '#40869c', test: s => s === 8 },
  { id: 'D', name: 'Верхний ряд', color: '#a88630', test: s => s < WIDTH },
  { id: 'E', name: 'Правая зона', color: '#69834a', test: s => s % WIDTH >= 2 },
];
export const DEFAULT_CONFIG = Object.freeze({
  enabled: [true, true, true, true, true], command: 0, seed: 1724,
  horizon: 8, iterations: 5, epochs: 3, gamma: .9, terminalGamma: .9,
  lambda: .95, cap: 8, criticLR: .8, actorLR: .6, valueLR: .4,
  clip: .2, beta: .4, entropy: .01, valueCoefficient: .5, danger: true,
});
export const BOUNDS = {
  seed: [0, 999999], horizon: [3, 10], iterations: [2, 8], epochs: [1, 5],
  gamma: [.5, .99], terminalGamma: [0, .99], lambda: [0, 1], cap: [1, 10],
  criticLR: [.01, 2], actorLR: [.01, 2], valueLR: [.01, 2],
  clip: [.05, .5], beta: [0, 2], entropy: [0, .1], valueCoefficient: [0, 1],
};
const clone = value => structuredClone(value);
const zeros = n => Array(n).fill(0);
const matrix = (n, m) => Array.from({ length: n }, () => zeros(m));
const tensor = (n, m, k) => Array.from({ length: n }, () => matrix(m, k));
const indicator = (condition) => Number(condition);
export const coordinates = s => [s % WIDTH, Math.floor(s / WIDTH)];

export function normalizeConfig(input = {}) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) input = {};
  const config = clone(DEFAULT_CONFIG);
  for (const [key, [lo, hi]] of Object.entries(BOUNDS)) {
    if (typeof input[key] === 'number' && Number.isFinite(input[key])) config[key] = clamp(input[key], lo, hi);
  }
  for (const key of ['seed', 'horizon', 'iterations', 'epochs', 'cap']) config[key] = Math.round(config[key]);
  if (Array.isArray(input.enabled) && input.enabled.length === 5 && input.enabled.every(v => typeof v === 'boolean'))
    config.enabled = input.enabled.slice();
  if (typeof input.danger === 'boolean') config.danger = input.danger;
  config.command = Number.isInteger(input.command) && input.command >= 0 && input.command < 5 ? input.command : config.command;
  if (!config.enabled[config.command]) config.command = config.enabled.indexOf(true);
  return config;
}

export function move(state, action, danger = true) {
  const [x, y] = coordinates(state);
  const direction = ACTIONS[action];
  const nx = x + direction.dx, ny = y + direction.dy;
  const outside = nx < 0 || nx >= WIDTH || ny < 0 || ny >= HEIGHT;
  const blocked = outside || ny * WIDTH + nx === WALL;
  const next = blocked ? state : ny * WIDTH + nx;
  return { state, action, next, blocked, fatal: danger && next === DANGER, attempted: [nx, ny] };
}

export function semantics(state, enabled, fatal = false) {
  const predicates = GOALS.map((g, i) => !fatal && enabled[i] && g.test(state));
  const count = predicates.filter(Boolean).length;
  const labels = predicates.map(p => p ? 1 / count : 0);
  return { predicates, count, labels, residual: count === 0 ? 1 : 0 };
}

export function initialize(config) {
  const random = rng(config.seed + 1000003);
  const parameters = {
    phi: tensor(12, 4, 2), psi: matrix(5, 2),
    theta: tensor(12, 5, 4), value: matrix(12, 5),
  };
  parameters.phi = parameters.phi.map(row => row.map(vector => vector.map(() => (random() - .5) * .6)));
  parameters.psi = parameters.psi.map(vector => vector.map(() => (random() - .5) * .6));
  // Нулевые policy logits дают ровно 1/4. Критик начинается ненулевым:
  // если оба множителя φ и ψ нулевые, градиенты факторизации тоже нулевые.
  return parameters;
}

export function criticForward(parameters, state, action, enabled) {
  const phi = parameters.phi[state][action];
  const products = parameters.psi.map(psi => psi.map((x, j) => x * phi[j]));
  const scores = products.map(sum);
  const maximum = Math.max(0, ...scores.filter((_, g) => enabled[g]));
  const shiftedExp = scores.map((score, g) => enabled[g] ? Math.exp(score - maximum) : 0);
  const absorbingExp = Math.exp(-maximum);
  const shiftedZ = sum(shiftedExp) + absorbingExp;
  const logZ = maximum + Math.log(shiftedZ);
  const probabilities = [...shiftedExp.map(x => x / shiftedZ), absorbingExp / shiftedZ];
  const logProbabilities = [...scores.map((f, g) => enabled[g] ? f - logZ : 0), -logZ];
  return { phi: phi.slice(), products, scores, maximum, shiftedExp, absorbingExp, shiftedZ, logZ,
    probabilities, logProbabilities };
}

export function collect(parameters, config, iteration) {
  const random = rng(config.seed + iteration * 997);
  const steps = [];
  let state = START;
  for (let t = 0; t < config.horizon; t++) {
    const logits = config.command < 0 ? zeros(4) : parameters.theta[state][config.command].slice();
    const policy = softmax(logits);
    const u = random();
    const cdf = policy.map((_, a) => sum(policy.slice(0, a + 1)));
    const action = Math.max(0, cdf.findIndex(p => u < p));
    const transition = move(state, action, config.danger);
    const semantic = semantics(transition.next, config.enabled, transition.fatal);
    const reward = config.command < 0 ? 0 : indicator(semantic.predicates[config.command]);
    const truncated = !transition.fatal && t === config.horizon - 1;
    const value = config.command < 0 ? 0 : parameters.value[state][config.command];
    const nextValue = config.command < 0 ? 0 : parameters.value[transition.next][config.command];
    steps.push({ ...transition, t, semantic, reward, truncated, logits, policy, u, cdf, value, nextValue });
    state = transition.next;
    if (transition.fatal) break;
  }
  return steps;
}

export function buildTarget(steps, anchor, config) {
  const terminal = steps.findIndex((s, i) => i >= anchor && s.fatal);
  const end = terminal < 0 ? steps.length : terminal;
  const distance = terminal < 0 ? null : terminal - anchor;
  const retained = distance === null ? 1 : 1 - config.terminalGamma ** distance;
  const futures = [];
  for (let i = anchor; i < Math.min(end, anchor + config.cap); i++) {
    const k = i - anchor + 1;
    futures.push({ t: i, next: steps[i].next, k, power: config.gamma ** (k - 1),
      raw: (1 - config.gamma) * config.gamma ** (k - 1), semantic: steps[i].semantic });
  }
  const normalizer = sum(futures.map(f => f.raw));
  const mean = zeros(5);
  for (const future of futures) {
    future.weight = future.raw / normalizer;
    future.contribution = future.semantic.labels.map(x => x * future.weight);
    future.corrected = future.contribution.map(x => x * retained);
    future.contribution.forEach((v, g) => { mean[g] += v; });
  }
  const goals = mean.map(x => x * retained);
  const absorbing = clamp(1 - sum(goals), 0, 1);
  return { futures, normalizer, distance, retained, mean, y: [...goals, absorbing],
    missingSemantic: retained * (1 - sum(mean)), terminalMass: 1 - retained };
}

export function criticUpdate(before, steps, config) {
  const phiGradient = tensor(12, 4, 2), psiGradient = matrix(5, 2);
  const count = steps.length;
  const anchors = steps.map((step, t) => {
    const target = buildTarget(steps, t, config);
    const forward = criticForward(before, step.state, step.action, config.enabled);
    const terms = target.y.map((y, g) => -y * forward.logProbabilities[g]);
    const scoreGradient = forward.probabilities.slice(0, 5).map((p, g) => config.enabled[g] ? p - target.y[g] : 0);
    const localPhi = [0, 1].map(j => sum(scoreGradient.map((v, g) => v * before.psi[g][j])));
    const localPsi = scoreGradient.map(v => forward.phi.map(x => v * x));
    for (let j = 0; j < 2; j++) phiGradient[step.state][step.action][j] += localPhi[j] / count;
    for (let g = 0; g < 5; g++) for (let j = 0; j < 2; j++) psiGradient[g][j] += localPsi[g][j] / count;
    return { target, forward, terms, loss: sum(terms), scoreGradient, localPhi, localPsi };
  });
  const after = clone(before);
  after.phi = before.phi.map((row, s) => row.map((vector, a) => vector.map((x, j) => x - config.criticLR * phiGradient[s][a][j])));
  after.psi = before.psi.map((vector, g) => vector.map((x, j) => x - config.criticLR * psiGradient[g][j]));
  const loss = sum(anchors.map(a => a.loss)) / count;
  const lossAfter = sum(steps.map((s, t) => {
    const forward = criticForward(after, s.state, s.action, config.enabled);
    return -sum(anchors[t].target.y.map((y, g) => y * forward.logProbabilities[g]));
  })) / count;
  return { anchors, phiGradient, psiGradient, after, loss, lossAfter };
}

export function advantages(steps, config) {
  let following = 0;
  const result = [];
  for (let t = steps.length - 1; t >= 0; t--) {
    const step = steps[t];
    const bootstrap = indicator(!step.fatal);
    const continuation = indicator(!step.fatal && !step.truncated && t + 1 < steps.length);
    const delta = step.reward + config.gamma * bootstrap * step.nextValue - step.value;
    const trace = config.gamma * config.lambda * continuation * following;
    const advantage = delta + trace;
    result[t] = { bootstrap, continuation, delta, trace, following, advantage, return: advantage + step.value };
    following = advantage;
  }
  return result;
}

export function teacher(parameters, state, config) {
  if (config.command < 0) return { scores: zeros(4), action: null };
  const scores = ACTIONS.map((_, a) => criticForward(parameters, state, a, config.enabled).scores[config.command]);
  return { scores, action: scores.indexOf(Math.max(...scores)) };
}

/** Локальные производные до усреднения по rollout. A, R, πold и teacher заморожены. */
export function policyTerms(logits, value, step, gae, teacherAction, beta, config) {
  const policy = softmax(logits);
  const logPolicy = policy.map(Math.log);
  const ratio = policy[step.action] / step.policy[step.action];
  const bounded = clamp(ratio, 1 - config.clip, 1 + config.clip);
  const raw = ratio * gae.advantage;
  const clipped = bounded * gae.advantage;
  const clippedActive = clipped < raw;
  const entropy = -sum(policy.map((p, a) => p * logPolicy[a]));
  const policyGradient = policy.map((p, a) => clippedActive ? 0 : -gae.advantage * ratio * (indicator(a === step.action) - p));
  const bcGradient = policy.map((p, a) => beta * (p - indicator(a === teacherAction)));
  const entropyGradient = policy.map((p, a) => config.entropy * p * (logPolicy[a] + entropy));
  const gradient = policyGradient.map((v, a) => v + bcGradient[a] + entropyGradient[a]);
  const policyLoss = -Math.min(raw, clipped);
  const bcLoss = -logPolicy[teacherAction];
  const valueLoss = .5 * (value - gae.return) ** 2;
  const valueGradient = config.valueCoefficient * (value - gae.return);
  const loss = policyLoss + beta * bcLoss + config.valueCoefficient * valueLoss - config.entropy * entropy;
  return { logits: logits.slice(), value, policy, logPolicy, ratio, bounded, raw, clipped, clippedActive,
    entropy, policyGradient, bcGradient, entropyGradient, gradient, policyLoss, bcLoss, valueLoss, valueGradient, loss };
}

export function studentUpdate(before, steps, gae, teachers, config, beta) {
  let current = clone(before);
  const epochs = [];
  if (config.command < 0) return { epochs, after: current };
  for (let epoch = 0; epoch < config.epochs; epoch++) {
    const epochBefore = clone(current);
    const thetaGradient = tensor(12, 5, 4), valueGradient = matrix(12, 5);
    const anchors = steps.map((step, t) => {
      const local = policyTerms(current.theta[step.state][config.command], current.value[step.state][config.command],
        step, gae[t], teachers[t].action, beta, config);
      for (let a = 0; a < 4; a++) thetaGradient[step.state][config.command][a] += local.gradient[a] / steps.length;
      valueGradient[step.state][config.command] += local.valueGradient / steps.length;
      return local;
    });
    // Обновления синхронные: все локальные производные используют один snapshot.
    current.theta = current.theta.map((row, s) => row.map((vector, g) => vector.map((x, a) => x - config.actorLR * thetaGradient[s][g][a])));
    current.value = current.value.map((row, s) => row.map((x, g) => x - config.valueLR * valueGradient[s][g]));
    epochs.push({ index: epoch, before: epochBefore, after: clone(current), thetaGradient, valueGradient, anchors,
      loss: sum(anchors.map(a => a.loss)) / steps.length });
  }
  return { epochs, after: current };
}

export function runIteration(previous, input, index = 0) {
  const config = normalizeConfig(input);
  const before = clone(previous);
  const steps = collect(before, config, index);
  const critic = criticUpdate(before, steps, config);
  const gae = advantages(steps, config);
  const teachers = steps.map(s => teacher(critic.after, s.state, config));
  const beta = config.beta * (1 - index / (config.iterations - 1));
  const student = studentUpdate(critic.after, steps, gae, teachers, config, beta);
  return { index, config, before, steps, critic, gae, teachers, beta, student, after: clone(student.after),
    reward: sum(steps.map(s => s.reward)), fatal: steps.at(-1).fatal };
}

export function runExperiment(input, count) {
  const config = normalizeConfig(input);
  let parameters = initialize(config);
  const iterations = [];
  const limit = Math.min(config.iterations, Math.max(1, Math.round(count || 1)));
  for (let index = 0; index < limit; index++) {
    const iteration = runIteration(parameters, config, index);
    iterations.push(iteration);
    parameters = iteration.after;
  }
  return { config, iterations };
}
