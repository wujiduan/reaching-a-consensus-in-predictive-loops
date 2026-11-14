import { DEFAULTS, normalizeConfig, simulate } from './engine.js';
import { drawNetwork, drawDistribution, drawTrajectories, drawDisagreement, hitTestNetwork } from './visuals.js';

const $ = id => document.getElementById(id);
const all = selector => [...document.querySelectorAll(selector)];
const BASE = { ...DEFAULTS, density: .16, mixing: .02, spread: .08, alignment: 'community', beta: .85 };
const SCENARIOS = {
  classic: { title: 'Without a platform', description: 'Innate opinions anchor each peer interaction. With β = 0, retraining adds no feedback: every round returns to the same finite-K FJ outcome. Compare it with the dashed, fully equilibrated FJ reference.', config: { ...BASE, beta: 0 } },
  feedback: { title: 'The predictive loop', description: 'Repeated predictions feed expressed opinions back into the next round. Increase platform influence toward 1 and watch how opinions move closer together, even though the network and innate opinions stay fixed.', config: { ...BASE } },
  bridge: { title: 'Beyond social ties', description: 'These two communities have no social ties between them. Mean extrapolation creates a route for influence through the platform. The unobserved group ignores peers (α = 0). Inspect one of its members to see their response to the mean prediction.', config: { ...BASE, topology: 'disconnected', model: 'mean', observed: .5, sampling: 'community', heterogeneity: 'stubborn', stubbornFraction: .5, beta: 1, rounds: 200 } },
  nudge: { title: 'A targeted nudge', description: 'One person receives a fixed prediction of 1; everyone else receives their last expressed opinion. The intervention changes a prediction, not the person’s expressed opinion. The receiver has β = 0. Change β for the rest of the population to test how repeated learning amplifies the receiver’s response.', config: { ...BASE, n: 40, topology: 'complete', distribution: 'normal', mean: 0, spread: 0, alpha: .6, beta: .9, k: 0, model: 'intervention', targetNode: 0, targetBeta: .9, receiverNode: 1, target: 1 } }
};
const TOPOLOGY_NAMES = { communities: 'two communities', random: 'random network', smallworld: 'small-world network', scaleFree: 'preferential attachment', ring: 'ring', complete: 'fully connected', disconnected: 'disconnected communities' };
const MODEL_NAMES = { perfect: 'Perfect prediction', mean: 'Observed-mean extrapolation', linear: 'Linear regression', intervention: 'Targeted intervention' };
const POLICY_HELP = {
  perfect: 'The platform echoes each person’s latest expressed opinion.',
  mean: 'Observed people keep their own last opinion as a prediction. Unobserved people receive the observed group’s mean.',
  linear: 'Fit fixed synthetic profile features to observed opinions; predict for everyone and clip outputs to [0, 1].',
  intervention: 'Fix one person’s prediction at s. Echo the latest expressed opinion for everyone else.'
};
const NETWORK_HELP = {
  communities: 'Random ties within two groups, with a separate probability for bridges between them.',
  random: 'Each pair connects with the selected probability. Joining edges are added if needed for connectivity.',
  smallworld: 'Begin with a regular ring lattice, then rewire ties. Joining edges ensure connectivity.',
  scaleFree: 'New people connect preferentially to existing people with higher degree (attachment weight = degree + 1).',
  ring: 'Everyone has two neighbors. A sparse, regular graph where influence travels locally.',
  complete: 'Everyone is connected to everyone else. A regular graph with maximum mixing.',
  disconnected: 'Two internally connected groups with no ties between them. An exploratory departure from the usual connected-network assumption.'
};
const DATA_HELP = {
  bimodal: 'Two normal groups, centered at μ ± d/2. Samples outside [0, 1] are clipped. These are synthetic data, not the paper’s datasets.',
  normal: 'Normal samples with the chosen center and standard deviation, clipped to [0, 1]. Clipping can shift the realized mean.',
  uniform: 'Each innate opinion is drawn uniformly from [0, 1]. The resulting sample is reproducible with the random seed.',
  skewed: 'A Beta(2, 5) distribution concentrates innate opinions toward the low end. Its population mean is 2/7.',
  custom: 'Each value becomes one person’s fixed innate opinion. The number of people follows the number of values you provide.'
};
const METHODS = {
  source: 'https://arxiv.org/html/2603.12137v2', algorithm: 'Algorithm 1; equations 1–3, 5, 8, 10',
  update: ['Initialize f = Algo(xstar)', 'For each round: z_i = (1-beta_i)*xstar_i + beta_i*f_i; x = z', 'Repeat K times simultaneously: x_i = (1-alpha_i)*z_i + alpha_i*mean(x_neighbors)', 'Record expressed x; retrain f = Algo(x)'],
  kZero: 'K=0 is a UI sentinel for exact FJ equilibrium via an LU linear solve; it does not mean zero peer steps.',
  finiteK: 'Direct iteration or its algebraically equivalent matrix-power calculation for dense workloads.',
  baseline: 'Both references use beta=0 with the same graph, innate opinions, and alpha. baseline uses K=∞; sameKBaseline uses the current K.',
  equilibrium: 'For affine policies, solve the outer fixed point separately from the simulation horizon. For closed classes, the stationary average of xstar selects the limit from this initialization. Clipped regression reports a next-round residual.',
  partialMean: 'Fixed observed set. Copy observed opinions; use mean(observed opinions) for every unobserved person.',
  intervention: 'Clamp one prediction, not the expressed opinion. All other predictions copy their corresponding expressed opinion.',
  syntheticFeatures: ['intercept = 1', 'community = 2*group - 1', 'noisy innate metadata = 2*clip(xstar + Normal(0, 0.2), 0, 1) - 1'],
  linearFit: 'Observed-set squared-error regression; slopes penalized with ridge lambda, intercept unpenalized; predictions clipped [0,1]. Lambda=0 uses a pseudoinverse if rank deficient. Feature metadata stays fixed throughout retraining.',
  observations: 'Random sampling is a fixed seeded permutation. Community sampling selects IDs from the first group before the second. Exported world.observed is the effective mask; sampledObservationMask retains the subset that perfect/intervention policies ignore.',
  susceptibilities: 'Uniform: same alpha and beta. Heterogeneous: independent Normal(control value, 0.1) draws clipped to alpha [0,.99], beta [0,1]. Stubborn subgroup: last round(n*fraction) people have alpha=0.',
  graph: 'Undirected simple graphs. W is row-normalized adjacency. Connectivity is enforced by minimal joining edges, within each group for disconnected mode. Isolate convention is self-average.',
  preferentialAttachment: 'Attachment probability proportional to existing degree + 1; the density control sets m, capped at 12.',
  data: 'Normal and bimodal samples are clipped [0,1]. Bimodal centers are mean ± separation/2. Uniform is U[0,1], skewed is Beta(2,5).',
  randomness: 'Independent seeded streams for graph, innate data, assignment, metadata, susceptibilities, observation sample, and layout.',
  metrics: { variance: 'Population variance (denominator n)', disagreement: 'Mean squared difference over undirected edges', delta: 'Maximum absolute change from the previous expressed state', predictionError: 'Mean squared difference between deployed predictions and expressed opinions at that round', stabilityTolerance: 1e-7, nearConsensusRange: .01 },
  roundZero: 'Innate opinions, before the first platform-influence step; f contains initial fitted predictions.',
  limitations: 'Synthetic exploration, not a reproduction of Yelp/Pokec. Finite-horizon stability is a numerical criterion, not a theorem about arbitrary parameters.'
};

let config = { ...BASE }, run = null, round = 120, selected = -1, hovered = -1;
let scenario = 'feedback', pinned = null, playing = false, lastFrame = 0, frameId = 0;
let debounceTimer, toastTimer, worker = null, requestId = 0, computing = false;
let viewingEquilibrium = false, mechanismPerson = -1;
const numericKeys = new Set(Object.keys(DEFAULTS).filter(key => typeof DEFAULTS[key] === 'number'));

function safeText(element, text) { if (element) element.textContent = text; }
function toast(message) { safeText($('toast'), message); $('toast').classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 3700); }
function fixed(n, digits=3) { return Number.isFinite(n) ? n.toFixed(digits) : '—'; }
function varianceText(n) { return n > 0 && n < .0001 ? n.toExponential(1) : fixed(n,4); }
function smallValue(n) { return n > 0 && n < .001 ? '<0.001' : fixed(n); }
function setRangeFill(el) { const min=Number(el.min)||0,max=Number(el.max)||1; el.style.setProperty('--range-fill', `${100*(Number(el.value)-min)/(max-min || 1)}%`); }
function displayControl(key,value) {
  if (key==='observed'||key==='stubbornFraction') return `${Math.round(value*100)}%`;
  if (key==='rounds') return `${value} rounds`;
  if (key==='n'||key==='seed'||key==='targetNode') return String(value);
  if (key==='mixing') return value.toFixed(3).replace(/0$/,'');
  return Number(value).toFixed(2);
}
function syncControls() {
  for (const el of all('[data-config]')) {
    const key=el.dataset.config;
    if (config[key]===undefined) continue;
    if (el.tagName==='SELECT' && ![...el.options].some(o=>o.value===String(config[key]))) {
      const option=document.createElement('option');option.value=String(config[key]);option.textContent=`${config[key]} steps`;el.append(option);
    }
    if (el.type==='range') { el.max=String(Math.max(Number(el.max),config[key]));el.min=String(Math.min(Number(el.min),config[key])); }
    el.value=config[key];
    if (el.type==='range') setRangeFill(el);
    safeText($(`${key}-value`),displayControl(key,config[key]));
  }
  $('targetNode').max=String(config.n-1);
  $('receiverNode').max=String(config.n-1);
  $('n').disabled=config.distribution==='custom';
  if(config.customOpinions && config.distribution==='custom') $('custom-data').value=config.customOpinions.map(v=>Number(v.toFixed(6))).join(', ');
  updateConditionalControls();
}
function updateConditionalControls() {
  $('observation-controls').hidden=!['mean','linear'].includes(config.model);
  $('linear-controls').hidden=config.model!=='linear';
  $('intervention-controls').hidden=config.model!=='intervention';
  safeText($('model-help'),POLICY_HELP[config.model]);
  $('stubborn-controls').hidden=config.heterogeneity!=='stubborn';
  safeText($('heterogeneity-help'),config.heterogeneity==='uniform'?'Uniform susceptibility makes the feedback effect easier to isolate.':config.heterogeneity==='random'?'Independent normal draws with SD 0.1 around the sliders; α is clipped to [0, .99] and β to [0, 1].':'The highest-ID people have α = 0. They ignore peers but can still respond to the platform.');
  $('density-controls').hidden=['ring','complete'].includes(config.topology);
  $('mixing-controls').hidden=!['communities','smallworld'].includes(config.topology);
  safeText($('density-label'),['communities','disconnected'].includes(config.topology)?'Within-group edge probability':config.topology==='smallworld'?'Initial lattice density':config.topology==='scaleFree'?'Attachment density':'Edge probability');
  safeText($('mixing-label'),config.topology==='smallworld'?'Rewiring probability':'Between-group edge probability');
  safeText($('topology-help'),NETWORK_HELP[config.topology]);
  $('parametric-distribution').hidden=!['normal','bimodal'].includes(config.distribution);
  $('separation-controls').hidden=config.distribution!=='bimodal';
  $('alignment-controls').hidden=config.distribution!=='bimodal';
  $('custom-controls').hidden=config.distribution!=='custom';
  safeText($('distribution-note'),DATA_HELP[config.distribution]);
  document.querySelector('label[for="spread"]').textContent=config.distribution==='normal'?'Standard deviation':'Within-group standard deviation';
}
function markScenario(id) {
  scenario=id;
  for(const b of all('[data-preset]')) b.setAttribute('aria-pressed',String(b.dataset.preset===id));
  const info=SCENARIOS[id];
  safeText($('scenario-title'),info?.title || 'Your experiment');
  safeText($('scenario-description'),info?.description || 'Explore how a fixed social network and a changing predictor interact. Keep the seed unchanged to compare the effects of individual parameters, and pin a run before changing the setup.');
}
function initWorker() {
  try {
    worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{if(data.id!==requestId)return;if(data.error)simulationError(data.error);else receiveRun(data.result);};
    worker.onerror=()=>{worker?.terminate();worker=null;try{receiveRun(simulate(config));}catch(error){simulationError(error.message);}};
  } catch { worker=null; }
}
function setComputing(value) {
  computing=value;
  for (const id of ['play-button','step-button','rewind-button','compare-button','export-button','share-button','pin-run','download-json','download-csv','view-equilibrium','inspect-mechanism']) $(id).disabled=value;
  $('timeline').disabled=value;
}
function simulationError(message) { setComputing(false);$('fatal-error').hidden=false;safeText($('fatal-error'),`Could not run this experiment: ${message}`);safeText($('state-badge'),'Check inputs'); }
function compute() {
  pause();clearTimeout(debounceTimer);$('fatal-error').hidden=true;
  try {config=normalizeConfig(config);}catch(error){simulationError(error.message);return;}
  const id=++requestId;setComputing(true);
  safeText($('state-badge'),'Updating');
  if(worker)worker.postMessage({id,config});else try{receiveRun(simulate(config));}catch(error){simulationError(error.message);}
}
function receiveRun(result) {
  setComputing(false);
  run=result;config=result.config;round=config.rounds;selected=Math.min(selected,config.n-1);
  hovered=-1;
  $('node-tooltip').hidden=true;
  viewingEquilibrium=viewingEquilibrium&&Boolean(run.equilibrium);
  syncControls();$('timeline').max=String(config.rounds);$('timeline').value=String(round);
  safeText($('network-description'),`${config.n} people · ${run.world.edges.length} ties · ${TOPOLOGY_NAMES[config.topology]}`);
  $('inspect-node').replaceChildren(...Array.from({length:config.n},(_,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i}`;return option;}));
  $('inspect-node').value=String(Math.max(0,selected));
  updateNotes();updateEquilibrium();updateMechanism();render();updatePlayButton();
  if($('compare-dialog').open)renderComparison();
}
function updateNotes() {
  const notes=[];
  if(run.world.addedEdges)notes.push(`${run.world.addedEdges} joining edge${run.world.addedEdges===1?'':'s'} added ${config.topology==='disconnected'?'within the communities':'to keep the sampled network connected'}.`);
  if(run.world.componentCount>1)notes.push(`${run.world.componentCount} disconnected components. Global consensus is not guaranteed; the predictor may connect them indirectly.`);
  if(config.heterogeneity==='stubborn'&&config.stubbornFraction>0)notes.push('Peer-stubborn people (αᵢ = 0) fall outside the full-observation consensus theorem’s strictly positive α assumption.');
  if(config.topology==='scaleFree')notes.push(`Each newcomer makes ${run.world.attachmentCount} ties, sampled without replacement with weight degree + 1.`);
  if(config.topology==='smallworld')notes.push(`The initial ring lattice has ${run.world.latticeDegree} neighbors per person; rewiring probability is ${config.mixing}.`);
  if(['mean','linear'].includes(config.model))notes.push(`${run.world.observed.filter(Boolean).length} of ${config.n} people are observed. Hollow network nodes are unobserved.`);
  if(config.model==='linear')notes.push('Regression features: intercept, community membership, and a fixed noisy proxy for innate opinion. Predictions are clipped to [0, 1].');
  $('run-notes').replaceChildren(...notes.map(text=>{const p=document.createElement('p');p.textContent=text;return p;}));
}
function setMetrics(id, values) {
  $(id).replaceChildren(...values.map(([label, value]) => {
    const item = document.createElement('div');
    const term = document.createElement('dt');
    const definition = document.createElement('dd');
    term.textContent = label;
    definition.textContent = value;
    item.append(term, definition);
    return item;
  }));
}

function updateEquilibrium() {
  const eq = run.equilibrium;
  $('view-equilibrium').disabled = !eq;
  const description = eq
    ? (eq.closedClassCount ? 'Limit selected by these innate opinions.' : 'Fixed point of the retraining update.')
    : run.equilibriumIssue ? 'Fixed-point solve unavailable.' : 'Clipped regression · simulation endpoint.';
  safeText($('equilibrium-description'), description);
  if (eq) {
    const distance = Math.max(...eq.x.map((value, i) => Math.abs(value - run.final.x[i])));
    setMetrics('equilibrium-metrics', [
      ['Mean opinion', fixed(eq.mean)], ['Opinion variance', varianceText(eq.variance)],
      ['Edge disagreement', varianceText(eq.disagreement)], ['Opinion range', smallValue(eq.range)],
      ['Max gap from horizon', varianceText(distance)],
    ]);
    const agreement = eq.range < .01 ? 'Near consensus (range < 0.01).' : 'Disagreement persists at equilibrium.';
    safeText($('equilibrium-note'), agreement);
  } else {
    setMetrics('equilibrium-metrics', [['Max next-round change', varianceText(run.residual)]]);
    safeText($('equilibrium-note'), run.equilibriumIssue || 'Use the horizon control to check whether the trajectory continues to move.');
  }
}

function updateMechanism() {
  const { world, history, equilibrium, final } = run;
  const endpoint = equilibrium || final;
  mechanismPerson = -1;
  $('mechanism-card').hidden = true;
  if (config.model === 'mean') {
    mechanismPerson = world.observed.findIndex((seen, i) => !seen && world.alpha[i] === 0);
    if (mechanismPerson < 0) return;
    const i = mechanismPerson;
    safeText($('mechanism-title'), 'Spillover through mean prediction');
    safeText($('mechanism-description'), `Person #${i} is unobserved and ignores peers (α = 0). Without platform influence, their opinion equals their innate opinion.`);
    setMetrics('mechanism-metrics', [['No platform', fixed(world.xstar[i])], [equilibrium ? 'With platform at equilibrium' : 'With platform at horizon', fixed(endpoint.x[i])]]);
  } else if (config.model === 'intervention') {
    const i = config.receiverNode;
    if (i < 0 || i === config.targetNode) return;
    mechanismPerson = i;
    safeText($('mechanism-title'), 'Intervention spillover');
    safeText($('mechanism-description'), `Person #${i} has β = 0. Compare retraining with a platform that keeps its first predictions fixed.`);
    const frozen = history[1].x[i];
    const shift = endpoint.x[i] - frozen;
    setMetrics('mechanism-metrics', [['Fixed initial predictions', fixed(frozen, 4)], [equilibrium ? 'Retrained equilibrium' : 'Retrained at horizon', fixed(endpoint.x[i], 4)], ['Change due to retraining', `${shift > 0 ? '+' : ''}${fixed(shift, 4)}`]]);
  }
  $('mechanism-card').hidden = mechanismPerson < 0;
}

function activeFrame() {
  return viewingEquilibrium ? run.equilibrium : run.history[round];
}

function networkOptions(highlight) {
  return {
    selected: highlight,
    target: config.model === 'intervention' ? config.targetNode : -1,
    showObserved: ['mean','linear'].includes(config.model),
  };
}

function render() {
  if(!run||computing)return;
  const h=activeFrame(),initial=run.initialMetrics;
  const innate = round === 0 && !viewingEquilibrium;
  safeText($('metric-variance'),varianceText(h.variance));
  const change=initial.variance>1e-12?(1-h.variance/initial.variance)*100:null;
  safeText($('variance-change'),change===null?'from zero':`${change>=0?'↓':'↑'} ${Math.abs(change).toFixed(0)}%`);
  $('variance-change').style.color=change!==null&&change<0?'#ad7352':'';
  $('variance-change').style.background=change!==null&&change<0?'#f4e7d8':'';
  safeText($('variance-note'),innate?'The innate distribution':'Compared with innate opinions');
  safeText($('metric-mean'),fixed(h.mean));$('mean-marker').style.left=`${h.mean*100}%`;
  const meanDelta=h.mean-initial.mean;
  safeText($('mean-note'),`${meanDelta>=0?'+':''}${fixed(meanDelta)} from innate mean`);
  safeText($('metric-range'),smallValue(h.range));
  safeText($('range-note'),`${fixed(h.min)} to ${fixed(h.max)} · min / max`);
  const near=h.range<.01;
  const stable=viewingEquilibrium || (round>=2 && (round===config.rounds ? run.converged : h.delta<1e-7));
  const stateLabel = viewingEquilibrium ? 'Equilibrium' : innate ? 'Innate opinions' : `Round ${round}`;
  safeText($('state-badge'),innate?'Initial state':near?'Near consensus':stable?'Stable, diverse':'Still evolving');
  $('state-badge').classList.toggle('consensus',near&&!innate);
  $('state-badge').title=`Opinion range = ${h.range.toPrecision(5)}. Near-consensus threshold: range < 0.01.`;
  safeText($('round-display'),stateLabel.toUpperCase());
  safeText($('timeline-label'),viewingEquilibrium ? 'Viewing equilibrium' : `Round ${round} / ${config.rounds}`);
  $('timeline').value=String(round);setRangeFill($('timeline'));
  $('view-equilibrium').setAttribute('aria-pressed', String(viewingEquilibrium));
  safeText($('view-equilibrium'), viewingEquilibrium ? 'Back to timeline' : 'View equilibrium');
  if(innate)safeText($('distribution-insight'),'The innate distribution before the first platform interaction.');
  else if(initial.variance>1e-12)safeText($('distribution-insight'),`${Math.abs(change).toFixed(1)}% ${change>=0?'less':'more'} opinion variance than the innate distribution.`);
  else safeText($('distribution-insight'),`The innate opinions all agree. The current opinion range is ${fixed(h.range,4)}.`);
  const reference = $('baseline-mode').value;
  const showBaseline = reference !== 'off';
  const baseline = reference === 'full' ? run.baseline : run.sameKBaseline;
  const baselineMetrics = reference === 'full' ? run.baselineMetrics : run.sameKBaselineMetrics;
  const measure = $('disagreement-measure').value;
  all('.baseline-legend').forEach(el=>el.hidden=!showBaseline);
  drawNetwork($('network-canvas'),run.world,h.x,networkOptions(hovered>=0?hovered:selected));
  drawDistribution($('distribution-canvas'),h.x,run.world.xstar,baseline,{showBaseline});
  drawTrajectories($('trajectory-canvas'),run.history,round,{baselineMean:baselineMetrics.mean,showBaseline,selected});
  drawDisagreement($('variance-canvas'),run.history,round,{measure,baselineValue:baselineMetrics[measure],showBaseline});
  safeText($('trajectory-description'), selected >= 0 ? `Person #${selected} highlighted` : 'Each line is one person');
  safeText($('disagreement-description'), measure === 'variance' ? 'Variance across the population' : 'Mean squared difference across social ties');
  $('variance-canvas').setAttribute('aria-label',`Disagreement over retraining rounds. ${$('disagreement-description').textContent} at round ${round} is ${varianceText(run.history[round][measure])}.`);
  $('network-canvas').setAttribute('aria-label',`${config.n} people and ${run.world.edges.length} social ties. ${stateLabel}. Mean opinion ${fixed(h.mean)}; range ${fixed(h.min)} to ${fixed(h.max)}. Use Inspect a person below for individual values.`);
  $('distribution-canvas').setAttribute('aria-label',`${stateLabel} opinion distribution. Variance ${varianceText(h.variance)}; innate variance ${varianceText(initial.variance)}.`);
  updateInspector();
}
function updateInspector() {
  if(!run)return;const i=selected>=0?selected:Number($('inspect-node').value)||0;
  safeText($('selected-person-label'),selected>=0?`#${i}`:'');
  $('inspect-node').value=String(i);
  const h=activeFrame(),w=run.world;
  const values=[['Innate',fixed(w.xstar[i])],['Expressed',fixed(h.x[i])],[viewingEquilibrium?'Stable prediction':'Prediction',fixed(h.f[i])],['Peer α',fixed(w.alpha[i],2)],['Platform β',fixed(w.beta[i],2)],['Social ties',String(w.degree[i])],['No platform · same K',fixed(run.sameKBaseline[i])],['Long-run opinion',fixed(run.equilibrium?.x[i])],['Observed',['perfect','intervention'].includes(config.model)||w.observed[i]?'Yes':'No']];
  $('node-details').replaceChildren(...values.map(([label,value])=>{const div=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;div.append(dt,dd);return div;}));
}
function pause() {playing=false;cancelAnimationFrame(frameId);updatePlayButton();}
function updatePlayButton() {
  const button=$('play-button');button.querySelector('span').textContent=playing?'Pause':run&&round<config.rounds?'Play':'Replay';
  button.querySelector('svg').innerHTML=playing?'<path d="M5 4h3v12H5zM12 4h3v12h-3z"/>':'<path d="m6 4 10 6-10 6z"/>';
  button.setAttribute('aria-label',playing?'Pause playback':run&&round<config.rounds?'Play simulation':'Replay simulation from innate opinions');
}
function play() {
  if(!run)return;if(playing){pause();return;}viewingEquilibrium=false;if(round>=config.rounds)round=0;playing=true;lastFrame=0;updatePlayButton();
  const tick=time=>{if(!playing)return;const interval=115/Number($('speed').value);if(!lastFrame)lastFrame=time;if(time-lastFrame>=interval){const advance=Math.max(1,Math.floor((time-lastFrame)/interval));round=Math.min(config.rounds,round+advance);lastFrame=time;render();}if(round>=config.rounds){pause();return;}frameId=requestAnimationFrame(tick);};
  render();frameId=requestAnimationFrame(tick);
}
function setTab(name,focus=false) {
  for(const button of all('[data-tab]')){const active=button.dataset.tab===name;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;$(`panel-${button.dataset.tab}`).hidden=!active;if(active&&focus)button.focus();}
}
function changeSetting(el) {
  const key=el.dataset.config;
  const value=numericKeys.has(key)?Number(el.value):el.value;
  if(numericKeys.has(key)&&!Number.isFinite(value))return;
  config={...config,[key]:value};
  ++requestId;setComputing(true);
  if(key==='distribution'&&value==='custom'&&!config.customOpinions){config.customOpinions=run?.world.n===config.n?[...run.world.xstar]:Array(config.n).fill(.5);$('custom-data').value=config.customOpinions.map(v=>Number(v.toFixed(6))).join(', ');}
  if(key==='n'&&config.customOpinions&&config.customOpinions.length!==value)delete config.customOpinions;
  if(key==='distribution'&&value!=='custom')delete config.customOpinions;
  safeText($(`${key}-value`),displayControl(key,value));if(el.type==='range')setRangeFill(el);
  updateConditionalControls();markScenario(null);pause();
  safeText($('state-badge'),'Updating');
  clearTimeout(debounceTimer);debounceTimer=setTimeout(compute,el.type==='range'?110:0);
}
function renderComparison() {
  if(!run)return;const h=run.final,b=run.sameKBaselineMetrics,p=pinned?.final;
  const rows=[['Mean opinion',fixed(h.mean),fixed(p?.mean),fixed(b.mean)],['Opinion variance',varianceText(h.variance),p?varianceText(p.variance):'—',varianceText(b.variance)],['Opinion range',fixed(h.range,4),p?fixed(p.range,4):'—',fixed(b.range,4)],['Edge disagreement',varianceText(h.disagreement),p?varianceText(p.disagreement):'—',varianceText(b.disagreement)],['Equilibrium variance',run.equilibrium?varianceText(run.equilibrium.variance):'—',pinned?.equilibrium?varianceText(pinned.equilibrium.variance):'—',varianceText(b.variance)],['Equilibrium edge disagreement',run.equilibrium?varianceText(run.equilibrium.disagreement):'—',pinned?.equilibrium?varianceText(pinned.equilibrium.disagreement):'—',varianceText(b.disagreement)],['Round / horizon',String(config.rounds),pinned?String(pinned.config.rounds):'—','Same K'],['Max next-round change',run.residual.toExponential(2),pinned?pinned.residual.toExponential(2):'—','—']];
  const table=document.createElement('table');table.className='compare-table';
  const thead=document.createElement('thead'),head=document.createElement('tr');
  ['Metric','Current','Pinned run','No platform'].forEach(label=>{const th=document.createElement('th');th.scope='col';th.textContent=label;head.append(th);});thead.append(head);table.append(thead);
  const body=document.createElement('tbody');rows.forEach(values=>{const tr=document.createElement('tr');values.forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i===1)td.className='current';if(i===2&&!pinned)td.className='empty-cell';tr.append(td);});body.append(tr);});table.append(body);
  const details=document.createElement('div');details.className='compare-settings';
  details.textContent=`Current: ${MODEL_NAMES[config.model]} · α ${config.alpha}, β ${config.beta}, K ${config.k===0?'∞':config.k} · seed ${config.seed}. ${pinned?`Pinned: ${MODEL_NAMES[pinned.config.model]} · α ${pinned.config.alpha}, β ${pinned.config.beta}, K ${pinned.config.k===0?'∞':pinned.config.k} · seed ${pinned.config.seed}.`:'Pin the current experiment to start comparing.'}`;
  $('compare-content').replaceChildren(table,details);$('restore-pinned').disabled=!pinned;$('clear-pin').disabled=!pinned;
  safeText($('pin-run'),pinned?'Replace pinned run':'Pin current run');
}
function openDialog(id){pause();$(id).showModal();}
function download(content,type,name){const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);toast(`Downloaded ${name}`);}
function exportedRun(data){return{config:data.config,world:{n:data.world.n,edges:data.world.edges,neighbors:data.world.neighbors,degree:data.world.degree,groups:data.world.groups,positions:data.world.positions,xstar:data.world.xstar,alpha:data.world.alpha,beta:data.world.beta,observed:['perfect','intervention'].includes(data.config.model)?Array(data.world.n).fill(true):data.world.observed,sampledObservationMask:data.world.observed,features:data.world.features,components:data.world.components,addedEdges:data.world.addedEdges},history:data.history,baseline:data.baseline,baselineMetrics:data.baselineMetrics,sameKBaseline:data.sameKBaseline,sameKBaselineMetrics:data.sameKBaselineMetrics,equilibrium:data.equilibrium,equilibriumIssue:data.equilibriumIssue,final:data.final,finalPrediction:data.finalPrediction,residual:data.residual,convergenceRound:data.convergenceRound,convergenceTolerance:1e-7,consensusTolerance:.01};}
function exportJSON(){if(!run)return;const payload={schema:'predictive-loops-lab/1',createdAt:new Date().toISOString(),methods:METHODS,...exportedRun(run),pinned:pinned?exportedRun(pinned):null};download(JSON.stringify(payload,null,2),'application/json',`predictive-loops-seed-${config.seed}.json`);}
function exportCSV(){if(!run)return;const rows=['round,person,community,innate_opinion,expressed_opinion,deployed_prediction,peer_susceptibility,platform_susceptibility,observed,degree,full_fj_baseline,same_k_baseline'];const w=run.world;for(const h of run.history)for(let i=0;i<config.n;i++)rows.push([h.round,i,w.groups[i],w.xstar[i],h.x[i],h.f[i],w.alpha[i],w.beta[i],['perfect','intervention'].includes(config.model)||w.observed[i]?1:0,w.degree[i],run.baseline[i],run.sameKBaseline[i]].join(','));download(rows.join('\n'),'text/csv;charset=utf-8',`predictive-loops-seed-${config.seed}.csv`);}
async function share(){if(!run)return;const url=new URL(location.href);url.hash='exp='+encodeURIComponent(JSON.stringify(config));try{await navigator.clipboard.writeText(url.href);toast('Experiment link copied. It includes all settings and the seed.');}catch{const input=document.createElement('textarea');input.value=url.href;document.body.append(input);input.select();const success=document.execCommand('copy');input.remove();if(success)toast('Experiment link copied.');else{openDialog('export-dialog');toast('Clipboard unavailable. Export JSON to save this experiment.');}}}
function parseCustom(text){const tokens=text.trim().replace(/^\uFEFF/,'').split(/[,;\s]+/).filter(Boolean);if(tokens[0]&&/^"?opinion"?$/i.test(tokens[0]))tokens.shift();const values=tokens.map(token=>Number(token.replace(/^"|"$/g,'')));if(values.length<20||values.length>200)throw new Error(`Provide 20–200 opinions; found ${values.length}.`);if(values.some(v=>!Number.isFinite(v)||v<0||v>1))throw new Error('Each opinion must be a finite number from 0 to 1.');return values;}
function applyCustom(){try{const values=parseCustom($('custom-data').value);config={...config,distribution:'custom',n:values.length,customOpinions:values,targetNode:Math.min(config.targetNode,values.length-1)};safeText($('custom-error'),'');markScenario(null);compute();toast(`Loaded ${values.length} innate opinions.`);}catch(error){safeText($('custom-error'),error.message);}}
function loadConfig(input){config=normalizeConfig(input);viewingEquilibrium=false;markScenario(null);syncControls();compute();}

for(const el of all('[data-config]'))el.addEventListener(el.type==='range'?'input':'change',()=>changeSetting(el));
for(const button of all('[data-tab]')){
  button.addEventListener('click',()=>setTab(button.dataset.tab));
  button.addEventListener('keydown',event=>{const tabs=['dynamics','network','data'];let index=tabs.indexOf(button.dataset.tab);if(event.key==='ArrowRight')index=(index+1)%3;else if(event.key==='ArrowLeft')index=(index+2)%3;else if(event.key==='Home')index=0;else if(event.key==='End')index=2;else return;event.preventDefault();setTab(tabs[index],true);});
}
for(const button of all('[data-preset]'))button.addEventListener('click',()=>{const id=button.dataset.preset;config={...SCENARIOS[id].config};markScenario(id);syncControls();viewingEquilibrium=false;selected=-1;compute();toast(`Loaded “${SCENARIOS[id].title}”.`);});
$('reset-settings').addEventListener('click',()=>{config={...BASE};markScenario('feedback');viewingEquilibrium=false;selected=-1;syncControls();compute();toast('Default experiment restored.');});
$('play-button').addEventListener('click',play);
$('rewind-button').addEventListener('click',()=>{pause();viewingEquilibrium=false;round=0;render();updatePlayButton();});
$('step-button').addEventListener('click',()=>{if(!run)return;pause();viewingEquilibrium=false;round=round>=config.rounds?1:round+1;render();updatePlayButton();});
$('timeline').addEventListener('input',()=>{pause();viewingEquilibrium=false;round=Number($('timeline').value);render();updatePlayButton();});
$('baseline-mode').addEventListener('change',render);
$('disagreement-measure').addEventListener('change',render);
$('view-equilibrium').addEventListener('click',()=>{pause();viewingEquilibrium=!viewingEquilibrium;render();});
$('inspect-mechanism').addEventListener('click',()=>{
  selected=mechanismPerson;
  $('node-inspector').open=true;
  render();
  $('inspect-node').focus();
});
$('reshuffle').addEventListener('click',()=>{const array=new Uint32Array(1);crypto.getRandomValues(array);config.seed=1+array[0]%999999;markScenario(null);syncControls();compute();toast(`New random seed: ${config.seed}`);});
$('compare-button').addEventListener('click',()=>{if(!run)return;renderComparison();openDialog('compare-dialog');});
$('pin-run').addEventListener('click',()=>{pinned=structuredClone(run);renderComparison();toast('Current experiment pinned for comparison.');});
$('restore-pinned').addEventListener('click',()=>{if(pinned){loadConfig(pinned.config);$('compare-dialog').close();toast('Pinned settings restored.');}});
$('clear-pin').addEventListener('click',()=>{pinned=null;renderComparison();});
$('export-button').addEventListener('click',()=>{if(run)openDialog('export-dialog');});
$('download-json').addEventListener('click',exportJSON);$('download-csv').addEventListener('click',exportCSV);
$('share-button').addEventListener('click',share);
for(const button of all('.close-dialog'))button.addEventListener('click',()=>button.closest('dialog').close());
for(const dialog of all('dialog'))dialog.addEventListener('click',event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();}});
$('apply-custom').addEventListener('click',applyCustom);
$('custom-file').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;if(file.size>100000){safeText($('custom-error'),'Please use a file below 100 KB.');return;}$('custom-data').value=await file.text();applyCustom();event.target.value='';});
$('import-experiment').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>30000000)throw new Error('Please use a JSON file below 30 MB.');const payload=JSON.parse(await file.text());if(!payload.config||typeof payload.config!=='object')throw new Error('The file needs a valid experiment configuration.');loadConfig(payload.config);safeText($('import-error'),'');$('export-dialog').close();toast('Experiment settings imported and reproduced from the seed.');}catch(error){safeText($('import-error'),error.message);}event.target.value='';});
$('inspect-node').addEventListener('change',()=>{selected=Number($('inspect-node').value);render();});
$('node-inspector').addEventListener('toggle',()=>{if($('node-inspector').open&&selected<0){selected=0;render();}});
$('new-layout').addEventListener('click',()=>{if(!run)return;selected=(selected+1)%config.n;$('node-inspector').open=true;render();});
$('network-canvas').addEventListener('pointermove',event=>{
  if(!run||computing)return;const id=hitTestNetwork($('network-canvas'),run.world,event.clientX,event.clientY);if(hovered!==id){hovered=id;drawNetwork($('network-canvas'),run.world,activeFrame().x,networkOptions(id>=0?id:selected));}
  const tip=$('node-tooltip');tip.hidden=id<0;if(id<0)return;const h=activeFrame(),rect=$('network-canvas').getBoundingClientRect();tip.replaceChildren();const title=document.createElement('b');title.textContent=`Person ${id}`;const text=document.createElement('span');text.textContent=`Opinion ${fixed(h.x[id])} · ${run.world.degree[id]} ties`;tip.append(title,text);tip.style.left=`${Math.max(8,Math.min(rect.width-150,event.clientX-rect.left+12))}px`;tip.style.top=`${Math.max(38,Math.min(rect.height-55,event.clientY-rect.top-40))}px`;
});
$('network-canvas').addEventListener('pointerleave',()=>{hovered=-1;$('node-tooltip').hidden=true;render();});
$('network-canvas').addEventListener('click',event=>{if(!run||computing)return;const id=hitTestNetwork($('network-canvas'),run.world,event.clientX,event.clientY);if(id<0)return;selected=id;$('node-inspector').open=true;render();});
let resizeFrame;new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(render);}).observe(document.querySelector('.simulation'));
document.fonts?.ready.then(render);
window.addEventListener('hashchange',()=>{if(location.hash.startsWith('#exp=')){try{loadConfig(JSON.parse(decodeURIComponent(location.hash.slice(5))));toast('Experiment loaded from link.');}catch(error){toast(`Could not load this experiment: ${error.message}`);}}});

// Load reproducible state only from an explicit experiment fragment, never from page anchors.
if(location.hash.startsWith('#exp=')){try{config=normalizeConfig(JSON.parse(decodeURIComponent(location.hash.slice(5))));scenario=null;}catch(error){toast(`Invalid experiment link; using defaults. ${error.message}`);}}
$('n').step='1';$('rounds').step='1';
markScenario(scenario);syncControls();initWorker();compute();
