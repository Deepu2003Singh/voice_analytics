/* ===================================================
   VOXLENS — frontend logic
   =================================================== */

const $ = (id) => document.getElementById(id);

const dropzone   = $('dropzone');
const fileInput  = $('audio-input');
const fileLabel  = $('file-label');
const analyzeBtn = $('analyze-btn');
const uploadForm = $('upload-form');

const stateIdle    = $('state-idle');
const stateLoading = $('state-loading');
const stateError   = $('state-error');
const stateResults = $('state-results');

const statusDot  = $('status-dot');
const statusText = $('status-text');
const statusEl   = statusDot.parentElement;

let lastResult = null;

/* -------------------- Dropzone -------------------- */
dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); dropzone.classList.add('drag');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); dropzone.classList.remove('drag');
  })
);
dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    onFileSelected();
  }
});
fileInput.addEventListener('change', onFileSelected);

function onFileSelected() {
  if (fileInput.files.length) {
    fileLabel.textContent = fileInput.files[0].name;
    analyzeBtn.disabled = false;
  } else {
    fileLabel.textContent = '';
    analyzeBtn.disabled = true;
  }
}

/* -------------------- Status helpers -------------------- */
function setStatus(mode, text) {
  statusEl.className = 'status ' + mode;
  statusText.textContent = text;
}

function showState(state) {
  [stateIdle, stateLoading, stateError, stateResults].forEach(el => el.classList.add('hidden'));
  state.classList.remove('hidden');
}

/* -------------------- Submit -------------------- */
uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;

  showState(stateLoading);
  setStatus('busy', 'analysing…');

  // Rotate loading stage text
  const stages = [
    'Transcribing audio…',
    'Identifying speakers…',
    'Summarising conversation…',
    'Scoring sentiment…',
    'Extracting KPIs…',
  ];
  let idx = 0;
  $('loading-stage').textContent = stages[0];
  const stageTimer = setInterval(() => {
    idx = (idx + 1) % stages.length;
    $('loading-stage').textContent = stages[idx];
  }, 4500);

  const fd = new FormData();
  fd.append('audio', fileInput.files[0]);
  fd.append('context', uploadForm.context.value);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    clearInterval(stageTimer);

    if (!res.ok) {
      $('error-text').textContent = data.error || 'Unknown error';
      setStatus('err', 'error');
      showState(stateError);
      return;
    }

    lastResult = data;
    renderResults(data);
    setStatus('ok', 'analysis complete');
    showState(stateResults);
  } catch (err) {
    clearInterval(stageTimer);
    $('error-text').textContent = err.message;
    setStatus('err', 'network error');
    showState(stateError);
  }
});

/* -------------------- Renderers -------------------- */
function renderResults(d) {
  // Meta
  $('meta-file').textContent     = d.filename || '—';
  $('meta-duration').textContent = formatDuration(d.duration_seconds);
  $('meta-words').textContent    = d.word_count.toLocaleString();
  $('meta-context').textContent  = d.context.toUpperCase();

  // KPI cards
  const csat = d.kpis.customer_satisfaction;
  $('csat-score').textContent = csat.score;
  $('csat-label').textContent = csat.label;
  $('csat-bar').style.width   = csat.score + '%';

  const ap = d.kpis.agent_performance;
  $('agent-score').textContent = ap.score;
  $('agent-label').textContent = ap.label;
  $('agent-bar').style.width   = ap.score + '%';

  const s = d.sentiment.overall;
  const sentPct = Math.round(((s.compound + 1) / 2) * 100);
  $('sent-score').textContent = s.compound.toFixed(2);
  $('sent-label').textContent = s.label;
  $('sent-bar').style.width   = sentPct + '%';

  // Summary
  const ul = $('summary-bullets');
  ul.innerHTML = '';
  (d.summary.bullets.length ? d.summary.bullets : [d.summary.text || 'No summary.'])
    .forEach(b => {
      const li = document.createElement('li');
      li.textContent = b;
      ul.appendChild(li);
    });

  // Trajectory
  const traj = $('trajectory');
  traj.innerHTML = '';
  if (d.sentiment.trajectory.length === 0) {
    traj.innerHTML = '<div class="traj-cell neutral"><div class="traj-phase">N/A</div><div class="traj-label">No data</div></div>';
  } else {
    d.sentiment.trajectory.forEach(t => {
      const cell = document.createElement('div');
      cell.className = 'traj-cell ' + t.label.toLowerCase();
      cell.innerHTML = `
        <div class="traj-phase">${t.phase.toUpperCase()}</div>
        <div class="traj-label">${t.label}</div>
        <div class="traj-score">score: ${t.score}</div>
      `;
      traj.appendChild(cell);
    });
  }

  // Context KPIs
  const ctx = $('context-grid');
  ctx.innerHTML = '';
  const cs = d.kpis.context_specific || {};
  const cm = d.kpis.conversation_metrics;

  const items = [];
  // Always-on metrics
  items.push(['turns', cm.turn_count]);
  items.push(['agent talk ratio', ap.talk_ratio]);
  items.push(['filler words (agent)', ap.filler_word_count]);
  items.push(['words / minute', cm.words_per_minute]);

  // Context-specific
  if (d.context === 'sales') {
    items.push(['buying-intent signals', cs.buying_intent_signals ?? 0]);
    items.push(['objection signals', cs.objection_signals ?? 0]);
    items.push(['commitment secured', cs.commitment_secured ? 'YES' : 'NO']);
    items.push(['lead quality', cs.lead_quality ?? '—']);
  } else if (d.context === 'support') {
    items.push(['resolution signals', cs.resolution_signals ?? 0]);
    items.push(['escalation requested', cs.escalation_requested ? 'YES' : 'NO']);
    items.push(['likely resolved', cs.likely_resolved ? 'YES' : 'NO']);
    items.push(['top concerns', (cs.top_customer_concerns || []).slice(0, 3).join(', ') || '—']);
  }

  items.forEach(([k, v]) => {
    const el = document.createElement('div');
    el.className = 'ctx-item';
    let cls = '';
    if (v === 'YES') cls = 'good';
    if (v === 'NO') cls = 'bad';
    if (k === 'lead quality') cls = v === 'Hot' ? 'good' : v === 'Cold' ? 'bad' : 'warn';
    el.innerHTML = `<span class="ctx-key">${k}</span><span class="ctx-val ${cls}">${v}</span>`;
    ctx.appendChild(el);
  });

  // Keywords
  const kw = $('keywords');
  kw.innerHTML = '';
  d.kpis.top_keywords.forEach(w => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = w;
    kw.appendChild(chip);
  });

  // Transcript
  const tEl = $('transcript');
  tEl.innerHTML = '';
  d.segments.forEach(seg => {
    const div = document.createElement('div');
    div.className = 'turn';
    const sentTag = seg.sentiment_score > 0.25 ? 'pos'
                   : seg.sentiment_score < -0.25 ? 'neg'
                   : '';
    const sentLabel = sentTag === 'pos' ? 'positive'
                     : sentTag === 'neg' ? 'negative'
                     : 'neutral';
    div.innerHTML = `
      <div class="turn-meta">
        <span class="speaker ${seg.speaker.toLowerCase()}">${seg.speaker}</span>
        <span>${formatTime(seg.start)} → ${formatTime(seg.end)}</span>
        <span class="sent-tag ${sentTag}">${sentLabel} · ${seg.sentiment_score.toFixed(2)}</span>
      </div>
      <div class="turn-text">${escapeHtml(seg.text)}</div>
    `;
    tEl.appendChild(div);
  });
}

/* -------------------- Utility -------------------- */
function formatDuration(s) {
  if (!s) return '—';
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}
function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}
function escapeHtml(str) {
  return str.replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* -------------------- Buttons -------------------- */
$('copy-transcript').addEventListener('click', () => {
  if (!lastResult) return;
  navigator.clipboard.writeText(lastResult.transcript);
  const btn = $('copy-transcript');
  const orig = btn.textContent;
  btn.textContent = 'copied ✓';
  setTimeout(() => btn.textContent = orig, 1500);
});

$('download-json').addEventListener('click', () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `voxlens_${lastResult.job_id}.json`;
  a.click();
});

$('new-analysis').addEventListener('click', () => {
  fileInput.value = '';
  fileLabel.textContent = '';
  analyzeBtn.disabled = true;
  lastResult = null;
  setStatus('', 'system idle');
  showState(stateIdle);
});

/* -------------------- Boot -------------------- */
showState(stateIdle);
setStatus('', 'system idle');
