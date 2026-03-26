const apiBase = window.apiBase || 'http://localhost:8000'

const modeOptionsEl = document.getElementById('modeOptions')
const videoInput = document.getElementById('videoInput')
const previewVideo = document.getElementById('previewVideo')
const selectedFileNameEl = document.getElementById('selectedFileName')
const durationHintEl = document.getElementById('durationHint')

const runBtn = document.getElementById('runBtn')
const clearBtn = document.getElementById('clearBtn')
const toggleRawBtn = document.getElementById('toggleRawBtn')
const jumpAnalyzeBtn = document.getElementById('jumpAnalyzeBtn')

const statusEl = document.getElementById('status')
const summaryCardEl = document.getElementById('summaryCard')
const rawJsonEl = document.getElementById('rawJson')
const rawPanelEl = document.getElementById('rawPanel')

const riskGaugeCanvas = document.getElementById('riskGaugeChart')
const riskScoreValueEl = document.getElementById('riskScoreValue')
const riskLevelBadgeEl = document.getElementById('riskLevelBadge')

const scoreCanvas = document.getElementById('scoreChart')
const statusCanvas = document.getElementById('statusChart')
const distributionCanvas = document.getElementById('distributionChart')
const controlCanvas = document.getElementById('controlChart')
const videoTimelineCanvas = document.getElementById('videoTimelineChart')
const videoTimelineHintEl = document.getElementById('videoTimelineHint')
const videoFrameStripEl = document.getElementById('videoFrameStrip')
const acousticRadarCanvas = document.getElementById('acousticRadarChart')
const acousticTimelineCanvas = document.getElementById('acousticTimelineChart')
const acousticTimelineHintEl = document.getElementById('acousticTimelineHint')

const videoMarkerListEl = document.getElementById('videoMarkerList')
const acousticInsightsEl = document.getElementById('acousticInsights')
const transcriptTextEl = document.getElementById('transcriptText')
const textMarkerInfoEl = document.getElementById('textMarkerInfo')
const influenceTableWrapEl = document.getElementById('influenceTableWrap')

const pipelineStageEl = document.getElementById('pipelineStage')
const pipelineBarEl = document.getElementById('pipelineBar')
const pipelineSpinnerEl = document.getElementById('pipelineSpinner')
const pipelineNodes = {
  input: document.getElementById('nodeInput'),
  prep: document.getElementById('nodePrep'),
  models: document.getElementById('nodeModels'),
  fusion: document.getElementById('nodeFusion'),
  done: document.getElementById('nodeDone')
}

const MODE_CONFIGS = [
  { id: 'all', label: 'All Models (NLP + CV + CV+Audio)', endpoint: '/predict?include_cv=true&include_cv_audio=true&video_sample_every=8&cv_audio_sample_rate=0.25', method: 'POST', needsFile: true },
  { id: 'nlp', label: 'NLP only', endpoint: '/test/nlp', method: 'POST', needsFile: true },
  { id: 'cv', label: 'CV only', endpoint: '/test/cv?sample_every=8', method: 'POST', needsFile: true },
  { id: 'cv_audio', label: 'CV+Audio only', endpoint: '/predict/cv-audio?sample_rate=0.25', method: 'POST', needsFile: true },
  { id: 'deception', label: 'Deception Detection', endpoint: '/test/deception', method: 'POST', needsFile: true },
  { id: 'emotion_av', label: 'Emotion Audio+Video', endpoint: '/test/emotion-av?sample_every=8&sample_rate=0.25', method: 'POST', needsFile: true },
  { id: 'anomaly_audio', label: 'Anomaly Audio', endpoint: '/test/anomaly/audio', method: 'POST', needsFile: true },
  { id: 'anomaly_video', label: 'Anomaly Video', endpoint: '/test/anomaly/video?sample_every=8', method: 'POST', needsFile: true },
  { id: 'anomaly_text', label: 'Anomaly Text', endpoint: '/test/anomaly/text', method: 'POST', needsFile: true }
]

const charts = { gauge: null, score: null, status: null, distribution: null, control: null, videoTimeline: null, acousticRadar: null, acousticTimeline: null }
let previewUrl = null
let activeTermMap = new Map()
let pipelineTimer = null
let pipelineStepIndex = 0
let frameRenderToken = 0

const PIPELINE_STEPS = [
  { key: 'input', label: 'Input received', progress: 16 },
  { key: 'prep', label: 'Preprocessing media', progress: 38 },
  { key: 'models', label: 'Running models', progress: 67 },
  { key: 'fusion', label: 'Building multimodal fusion', progress: 88 },
  { key: 'done', label: 'Completed', progress: 100 }
]

const gaugeTextPlugin = {
  id: 'gaugeTextPlugin',
  afterDraw(chart) {
    const value = chart?.config?.options?.plugins?.gaugeTextPlugin?.value
    if(value === undefined || value === null) return
    const meta = chart.getDatasetMeta(0)
    if(!meta?.data?.length) return
    const center = meta.data[0]
    const ctx = chart.ctx
    ctx.save()
    ctx.fillStyle = '#deebff'
    ctx.font = '700 24px "Rajdhani", sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`${Math.round(Number(value) * 100)}%`, center.x, center.y + 12)
    ctx.restore()
  }
}

if(window.Chart && !Chart.registry.plugins.get('gaugeTextPlugin')) Chart.register(gaugeTextPlugin)

function setStatus(text, state) {
  statusEl.textContent = text
  if(state) statusEl.dataset.state = state
  else delete statusEl.dataset.state
}

function resetPipelineState() {
  if(pipelineTimer){
    clearInterval(pipelineTimer)
    pipelineTimer = null
  }
  pipelineStepIndex = 0
  pipelineStageEl.textContent = 'Idle'
  pipelineBarEl.style.width = '0%'
  pipelineBarEl.style.background = 'linear-gradient(90deg, #4aa8ff, #57d0c9)'
  Object.values(pipelineNodes).forEach(node => {
    if(!node) return
    node.classList.remove('is-running', 'is-complete')
  })
  pipelineSpinnerEl?.classList.remove('is-active')
}

function paintPipelineStep(stepIndex) {
  const safeIndex = Math.max(0, Math.min(stepIndex, PIPELINE_STEPS.length - 1))
  const step = PIPELINE_STEPS[safeIndex]
  pipelineStageEl.textContent = step.label
  pipelineBarEl.style.width = `${step.progress}%`

  Object.values(pipelineNodes).forEach(node => {
    if(!node) return
    node.classList.remove('is-running', 'is-complete')
  })

  PIPELINE_STEPS.forEach((item, idx) => {
    const node = pipelineNodes[item.key]
    if(!node) return
    if(idx < safeIndex) node.classList.add('is-complete')
    else if(idx === safeIndex) node.classList.add('is-running')
  })
}

function startPipelineAnimation() {
  resetPipelineState()
  pipelineSpinnerEl?.classList.add('is-active')
  paintPipelineStep(0)
  pipelineStepIndex = 1
  pipelineTimer = setInterval(() => {
    if(pipelineStepIndex >= PIPELINE_STEPS.length - 1){
      return
    }
    paintPipelineStep(pipelineStepIndex)
    pipelineStepIndex += 1
  }, 1250)
}

function finishPipeline(success = true) {
  if(pipelineTimer){
    clearInterval(pipelineTimer)
    pipelineTimer = null
  }
  pipelineSpinnerEl?.classList.remove('is-active')

  if(success){
    paintPipelineStep(PIPELINE_STEPS.length - 1)
    return
  }

  pipelineStageEl.textContent = 'Request failed'
  pipelineBarEl.style.width = '100%'
  pipelineBarEl.style.background = 'linear-gradient(90deg, #ef4444, #f59e0b)'
  Object.values(pipelineNodes).forEach(node => {
    if(!node) return
    node.classList.remove('is-running', 'is-complete')
  })
}

function escapeHtml(str) {
  return String(str).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
}

function escapeRegex(str) {
  return String(str).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function formatNumber(v) {
  if(v === null || v === undefined) return '-'
  const n = Number(v)
  if(Number.isNaN(n)) return String(v)
  if(n !== 0 && Math.abs(n) < 0.001) {
    const tiny = Math.round(n * 10000) / 10000
    return tiny === 0 ? '0' : tiny.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  }
  return (Math.round(n * 1000) / 1000).toString()
}

function clamp01(v) {
  const n = Number(v)
  if(Number.isNaN(n)) return 0
  return Math.max(0, Math.min(1, n))
}

function avg(values) {
  if(!values?.length) return 0
  return values.reduce((s, v) => s + v, 0) / values.length
}

function std(values) {
  if(!values || values.length <= 1) return 0
  const m = avg(values)
  return Math.sqrt(avg(values.map(v => (v - m) ** 2)))
}

function formatTime(seconds) {
  const n = Number(seconds)
  if(Number.isNaN(n) || n < 0) return '-'
  const mins = Math.floor(n / 60)
  const secs = Math.floor(n % 60)
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

function modelTitle(name) {
  const map = { nlp: 'NLP', cv: 'CV', video: 'CV', cv_audio: 'CV+Audio', deception: 'Deception', emotion_av: 'Emotion AV', anomaly_text: 'Anomaly text', anomaly_audio: 'Anomaly audio', anomaly_video: 'Anomaly video', ensemble: 'Ensemble' }
  return map[name] || name
}

function pickScore(obj) {
  if(!obj || typeof obj !== 'object') return null
  for(const key of ['probability', 'risk_score', 'deception_score', 'emotion_score', 'anomaly_score', 'ensemble_score']) {
    if(obj[key] !== undefined && obj[key] !== null && !Number.isNaN(Number(obj[key]))) return Number(obj[key])
  }
  return null
}

function normalizeResponse(json) {
  if(json && typeof json === 'object') {
    if(json.formatted_result) return json.formatted_result
    if(json.formatted) return json.formatted
  }
  return json
}

function buildModeOptions() {
  modeOptionsEl.innerHTML = ''
  MODE_CONFIGS.forEach((cfg, index) => {
    const label = document.createElement('label')
    label.innerHTML = `<input type="radio" name="mode" value="${cfg.id}" ${index === 0 ? 'checked' : ''}> ${cfg.label}`
    modeOptionsEl.appendChild(label)
  })
}

function getSelectedMode() {
  const selected = document.querySelector('input[name="mode"]:checked')
  return MODE_CONFIGS.find(m => m.id === selected?.value) || MODE_CONFIGS[0]
}

function extractModelRecords(data, modeCfg) {
  const records = []
  if(data?.models && typeof data.models === 'object') {
    Object.entries(data.models).forEach(([name, value]) => {
      if(!value || typeof value !== 'object') return
      records.push({ name, success: value.success, prediction: value.prediction, prediction_label: value.prediction_label, risk_level: value.risk_level, score: pickScore(value), error: value.error })
    })
  }
  if(data?.overall?.ensemble?.ensemble_score !== undefined) {
    records.push({ name: 'ensemble', success: true, prediction: data.overall.ensemble.ensemble_prediction, prediction_label: data.overall.ensemble.ensemble_prediction_label, risk_level: data.overall.ensemble.risk_level, score: Number(data.overall.ensemble.ensemble_score), error: null })
  }
  if(records.length === 0) {
    records.push({ name: modeCfg.id, success: data?.success, prediction: data?.prediction, prediction_label: data?.prediction_label, risk_level: data?.risk_level, score: pickScore(data), error: data?.error })
    if(data?.cv_audio_probability !== undefined) {
      records.push({ name: 'cv_audio', success: data?.cv_audio_success, prediction: null, prediction_label: null, risk_level: null, score: Number(data.cv_audio_probability), error: data?.cv_audio_error })
    }
  }
  const dedup = new Map()
  records.forEach(r => { if(r?.name && !dedup.has(r.name)) dedup.set(r.name, r) })
  return Array.from(dedup.values())
}

function extractTimeline(data) {
  if(Array.isArray(data?.time_series)) {
    return data.time_series.map((x, idx) => ({ x: x.start !== undefined ? Number(x.start) : idx, y: x.score !== undefined ? Number(x.score) : null, segment: x })).filter(p => p.y !== null && !Number.isNaN(p.y))
  }
  if(Array.isArray(data?.segments)) {
    return data.segments.map((seg, idx) => ({ x: seg.start_time !== undefined ? Number(seg.start_time) : (seg.start !== undefined ? Number(seg.start) : idx), y: seg.risk_score !== undefined ? Number(seg.risk_score) : (seg.score !== undefined ? Number(seg.score) : null), segment: seg })).filter(p => p.y !== null && !Number.isNaN(p.y))
  }
  return []
}

function downsampleTimeline(points, maxPoints = 180) {
  if(!Array.isArray(points) || points.length <= maxPoints) return points || []
  const step = Math.ceil(points.length / maxPoints)
  const compact = []
  for(let i = 0; i < points.length; i += step){
    compact.push(points[i])
  }
  if(compact[compact.length - 1] !== points[points.length - 1]){
    compact.push(points[points.length - 1])
  }
  return compact
}

function riskLevelFromScore(score) {
  const s = Number(score)
  if(Number.isNaN(s)) return 'unknown'
  if(s >= 0.7) return 'high'
  if(s >= 0.4) return 'medium'
  return 'low'
}

function pickOverallRisk(data, records) {
  const ensembleScore = data?.overall?.ensemble?.ensemble_score
  if(ensembleScore !== undefined && ensembleScore !== null && !Number.isNaN(Number(ensembleScore))) return Number(ensembleScore)
  const overallScore = pickScore(data?.overall || data)
  if(overallScore !== null) return Number(overallScore)
  const candidates = records.map(r => Number(r.score)).filter(v => !Number.isNaN(v))
  return candidates.length ? candidates[0] : null
}

function pickOverallLevel(data, score) {
  return data?.overall?.ensemble?.risk_level || data?.overall?.risk_level || data?.risk_level || riskLevelFromScore(score)
}

function pickRecordScore(records, preferredNames = []) {
  for(const name of preferredNames) {
    const record = records.find(r => r?.name === name && r.score !== null && r.score !== undefined && !Number.isNaN(Number(r.score)))
    if(record) return clamp01(Number(record.score))
  }
  return null
}

function getPreviewDurationSeconds() {
  const duration = Number(previewVideo?.duration)
  if(Number.isFinite(duration) && duration > 0) return duration
  return 60
}

function buildFlatTimeline(score, points = 18) {
  const safeScore = clamp01(score)
  const duration = getPreviewDurationSeconds()
  if(points <= 1) return [{ x: 0, y: safeScore }]
  return Array.from({ length: points }, (_, index) => {
    const ratio = index / (points - 1)
    return { x: duration * ratio, y: safeScore }
  })
}

function resolveTimeline(points, fallbackScore) {
  if(Array.isArray(points) && points.length > 0) {
    return { points, synthetic: false }
  }
  if(fallbackScore === null || fallbackScore === undefined || Number.isNaN(Number(fallbackScore))) {
    return { points: [], synthetic: false }
  }
  return { points: buildFlatTimeline(fallbackScore), synthetic: true }
}

function destroyChart(key) {
  if(charts[key]) { charts[key].destroy(); charts[key] = null }
}

function clearCharts() {
  Object.keys(charts).forEach(destroyChart)
}

function getChartContext(canvas, chartName) {
  if(!canvas) {
    console.warn(`Canvas not found for ${chartName}`)
    return null
  }
  const ctx = canvas.getContext('2d')
  if(!ctx) {
    console.warn(`2D context unavailable for ${chartName}`)
    return null
  }
  return ctx
}

function baseChartOptions(extra = {}) {
  const base = {
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#bfd2f2', font: { family: 'Space Grotesk' } } },
      tooltip: { titleFont: { family: 'Space Grotesk' }, bodyFont: { family: 'Space Grotesk' } }
    },
    scales: {
      x: { ticks: { color: '#9eb7dc' }, grid: { color: 'rgba(122, 165, 231, 0.12)' } },
      y: { beginAtZero: true, max: 1, ticks: { color: '#9eb7dc' }, grid: { color: 'rgba(122, 165, 231, 0.12)' } }
    }
  }
  if(extra.plugins) base.plugins = { ...base.plugins, ...extra.plugins }
  if(extra.scales) base.scales = { ...base.scales, ...extra.scales }
  return { ...base, ...extra, plugins: base.plugins, scales: base.scales }
}

function setRiskBadge(level) {
  const clean = (level || 'unknown').toString().toLowerCase()
  riskLevelBadgeEl.classList.remove('low', 'medium', 'high')
  if(clean === 'low' || clean === 'medium' || clean === 'high') riskLevelBadgeEl.classList.add(clean)
  riskLevelBadgeEl.textContent = clean.charAt(0).toUpperCase() + clean.slice(1)
}

function colorByRisk(score) {
  const s = Number(score)
  if(Number.isNaN(s)) return '#8ca5cb'
  if(s >= 0.7) return '#ef4444'
  if(s >= 0.4) return '#f59e0b'
  return '#22c55e'
}

function renderGauge(score) {
  destroyChart('gauge')
  const hasScore = score !== null && score !== undefined && !Number.isNaN(Number(score))
  const s = hasScore ? clamp01(score) : 0
  const ctx = getChartContext(riskGaugeCanvas, 'riskGaugeChart')
  if(!ctx) return
  charts.gauge = new Chart(ctx, {
    type: 'doughnut',
    data: { labels: ['Risk', 'Remaining'], datasets: [{ data: [s, Math.max(0, 1 - s)], backgroundColor: [colorByRisk(s), 'rgba(126, 154, 196, 0.22)'], borderWidth: 0, hoverOffset: 0 }] },
    options: { cutout: '78%', rotation: -90, circumference: 180, plugins: { legend: { display: false }, gaugeTextPlugin: { value: s } } }
  })
  riskScoreValueEl.textContent = hasScore ? formatNumber(s) : '-'
}

function renderSummary(data, modeCfg, records, score, level) {
  const rows = []
  const overall = data?.overall || data || {}
  rows.push(['Mode', modeCfg.label])
  rows.push(['Success', overall.success ?? data?.success])
  rows.push(['Prediction', overall.prediction_label ?? data?.prediction_label ?? overall.prediction ?? data?.prediction ?? '-'])
  rows.push(['Risk score', formatNumber(score)])
  rows.push(['Risk level', level ?? '-'])
  if(data?.segments_used !== undefined) rows.push(['Segments used', data.segments_used])
  if(Array.isArray(data?.segments)) rows.push(['Segments', data.segments.length])
  if(data?.overall?.ensemble?.models_used) rows.push(['Models used', data.overall.ensemble.models_used.join(', ')])

  let html = '<h3>Summary</h3><table class="kv">'
  rows.forEach(([k, v]) => { html += `<tr><td><b>${escapeHtml(k)}</b></td><td>${escapeHtml(v ?? '-')}</td></tr>` })
  html += '</table><h3 style="margin-top:12px;">Model metrics</h3><table class="model-table"><thead><tr><th>Model</th><th>Success</th><th>Pred</th><th>Risk</th><th>Score</th><th>Error</th></tr></thead><tbody>'
  records.forEach(r => {
    const cls = r.success === true ? 'model-ok' : (r.success === false ? 'model-bad' : '')
    html += `<tr><td>${escapeHtml(modelTitle(r.name))}</td><td class="${cls}">${escapeHtml(r.success ?? '-')}</td><td>${escapeHtml(r.prediction_label ?? r.prediction ?? '-')}</td><td>${escapeHtml(r.risk_level ?? '-')}</td><td>${escapeHtml(formatNumber(r.score))}</td><td>${escapeHtml(r.error ?? '-')}</td></tr>`
  })
  summaryCardEl.innerHTML = `${html}</tbody></table>`
}

function renderScoreChart(records) {
  destroyChart('score')
  const items = records.filter(r => r.score !== null && r.score !== undefined && !Number.isNaN(Number(r.score)))
  if(items.length === 0) return
  const ctx = getChartContext(scoreCanvas, 'scoreChart')
  if(!ctx) return

  charts.score = new Chart(ctx, {
    type: 'bar',
    data: { labels: items.map(i => modelTitle(i.name)), datasets: [{ label: 'Score / Probability', data: items.map(i => Number(i.score)), backgroundColor: ['#4aa8ff', '#3ec3bb', '#22c55e', '#f59e0b', '#ef4444', '#8ab4ff'] }] },
    options: baseChartOptions({ plugins: { legend: { display: false } } })
  })
}

function renderStatusChart(records) {
  destroyChart('status')
  const counts = { ok: 0, fail: 0, unknown: 0 }
  records.forEach(r => { if(r.success === true) counts.ok += 1; else if(r.success === false) counts.fail += 1; else counts.unknown += 1 })
  const values = [counts.ok, counts.fail, counts.unknown]
  if(values.reduce((a, b) => a + b, 0) === 0) return
  const ctx = getChartContext(statusCanvas, 'statusChart')
  if(!ctx) return

  charts.status = new Chart(ctx, {
    type: 'doughnut',
    data: { labels: ['OK', 'Fail', 'Unknown'], datasets: [{ data: values, backgroundColor: ['#22c55e', '#ef4444', '#64748b'] }] },
    options: { maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#bfd2f2' } } } }
  })
}

function renderDistributionChart(data, records) {
  destroyChart('distribution')
  const ctx = getChartContext(distributionCanvas, 'distributionChart')
  if(!ctx) return

  if(data?.emotion_distribution && Object.keys(data.emotion_distribution).length > 0) {
    const entries = Object.entries(data.emotion_distribution)
    charts.distribution = new Chart(ctx, {
      type: 'pie',
      data: { labels: entries.map(([k]) => k), datasets: [{ data: entries.map(([, v]) => Number(v)), backgroundColor: ['#4aa8ff', '#22c55e', '#f59e0b', '#ef4444', '#57d0c9', '#8ab4ff'] }] },
      options: { maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#bfd2f2' } } } }
    })
    return
  }

  if(data?.indicators && Object.keys(data.indicators).length > 0) {
    const entries = Object.entries(data.indicators)
    charts.distribution = new Chart(ctx, {
      type: 'bar',
      data: { labels: entries.map(([k]) => k), datasets: [{ data: entries.map(([, v]) => Number(v)), backgroundColor: '#57d0c9' }] },
      options: baseChartOptions({ plugins: { legend: { display: false } } })
    })
    return
  }

  if(data?.components && Object.keys(data.components).length > 0) {
    const entries = Object.entries(data.components)
    charts.distribution = new Chart(ctx, {
      type: 'radar',
      data: { labels: entries.map(([k]) => k), datasets: [{ label: 'Components', data: entries.map(([, v]) => Number(v)), borderColor: '#4aa8ff', backgroundColor: 'rgba(74, 168, 255, 0.22)' }] },
      options: { maintainAspectRatio: false, scales: { r: { beginAtZero: true, max: 1, grid: { color: 'rgba(122, 165, 231, 0.2)' }, angleLines: { color: 'rgba(122, 165, 231, 0.2)' }, pointLabels: { color: '#bfd2f2' }, ticks: { color: '#9eb7dc', backdropColor: 'transparent' } } }, plugins: { legend: { labels: { color: '#bfd2f2' } } } }
    })
    return
  }

  if(data?.overall?.ensemble?.individual_scores) {
    const entries = Object.entries(data.overall.ensemble.individual_scores)
    charts.distribution = new Chart(ctx, {
      type: 'polarArea',
      data: { labels: entries.map(([k]) => modelTitle(k)), datasets: [{ data: entries.map(([, v]) => Number(v)), backgroundColor: ['#4aa8ff', '#22c55e', '#f59e0b', '#ef4444', '#57d0c9'] }] },
      options: {
        maintainAspectRatio: false,
        scales: {
          r: {
            ticks: { display: false },
            grid: { display: false },
            angleLines: { display: false },
            pointLabels: { display: false }
          }
        },
        plugins: { legend: { position: 'bottom', labels: { color: '#bfd2f2' } } }
      }
    })
    return
  }

  const items = records.filter(r => r.score !== null && r.score !== undefined && !Number.isNaN(Number(r.score)))
  if(items.length === 0) return
  charts.distribution = new Chart(ctx, {
    type: 'doughnut',
    data: { labels: items.map(r => modelTitle(r.name)), datasets: [{ data: items.map(r => Number(r.score)), backgroundColor: ['#4aa8ff', '#57d0c9', '#22c55e', '#f59e0b', '#ef4444'] }] },
    options: { maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#bfd2f2' } } } }
  })
}

function renderVideoTimelineChart(timelineState) {
  destroyChart('videoTimeline')
  const timeline = timelineState?.points || []
  if(!timeline || timeline.length === 0) {
    if(videoTimelineHintEl) videoTimelineHintEl.textContent = 'No CV timeline available for this mode.'
    return
  }

  const sorted = timeline.map(p => ({ x: Number(p.x), y: Number(p.y) })).filter(p => !Number.isNaN(p.x) && !Number.isNaN(p.y)).sort((a, b) => a.x - b.x)
  if(sorted.length === 0) {
    if(videoTimelineHintEl) videoTimelineHintEl.textContent = 'CV timeline data is empty after filtering.'
    return
  }
  const compact = downsampleTimeline(sorted, 150)
  if(videoTimelineHintEl) {
    videoTimelineHintEl.textContent = timelineState?.synthetic
      ? 'Showing a model-level fallback line because frame-by-frame CV markers are unavailable.'
      : `Showing ${compact.length} timeline points.`
  }
  const ctx = getChartContext(videoTimelineCanvas, 'videoTimelineChart')
  if(!ctx) return

  charts.videoTimeline = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Risk trajectory',
        data: compact,
        parsing: false,
        borderColor: '#4aa8ff',
        backgroundColor: 'rgba(74, 168, 255, 0.22)',
        tension: 0.28,
        fill: true,
        pointRadius: 2
      }]
    },
    options: baseChartOptions({
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'linear',
          ticks: { color: '#9eb7dc' },
          grid: { color: 'rgba(122, 165, 231, 0.12)' },
          title: { display: true, text: 'time / index', color: '#9eb7dc' }
        },
        y: {
          beginAtZero: true,
          max: 1,
          ticks: { color: '#9eb7dc' },
          grid: { color: 'rgba(122, 165, 231, 0.12)' }
        }
      }
    })
  })
}

function renderVideoMarkers(data, timeline, records) {
  const segments = Array.isArray(data?.segments) ? data.segments : []
  const items = []

  if(segments.length) {
    const topSegments = [...segments].filter(s => s && s.risk_score !== undefined).sort((a, b) => Number(b.risk_score || 0) - Number(a.risk_score || 0)).slice(0, 3)
    topSegments.forEach(seg => {
      const hints = Array.isArray(seg.top_features) ? seg.top_features.slice(0, 2).map(f => f.feature || f.type).filter(Boolean) : []
      items.push({ title: `${formatTime(seg.start_time ?? seg.start)} - ${formatTime(seg.end_time ?? seg.end)}`, score: Number(seg.risk_score), note: hints.length ? `Top signals: ${hints.join(', ')}` : (seg.text ? String(seg.text).slice(0, 90) : 'No marker detail') })
    })
  }

  if(items.length === 0 && timeline.length) {
    const maxPoint = [...timeline].sort((a, b) => b.y - a.y)[0]
    items.push({ title: `Peak at ${formatTime(maxPoint.x)}`, score: maxPoint.y, note: 'Highest risk point on timeline.' })
  }

  if(items.length === 0) {
    const cv = records.find(r => r.name === 'cv')
    const cvAudio = records.find(r => r.name === 'cv_audio')
    if(cv?.score !== undefined) items.push({ title: 'CV marker score', score: Number(cv.score), note: `Prediction: ${cv.prediction_label ?? cv.prediction ?? '-'}` })
    if(cvAudio?.score !== undefined) items.push({ title: 'CV+Audio marker score', score: Number(cvAudio.score), note: `Prediction: ${cvAudio.prediction_label ?? cvAudio.prediction ?? '-'}` })
  }

  if(items.length === 0) {
    videoMarkerListEl.className = 'marker-list empty-note'
    videoMarkerListEl.textContent = 'No marker-level data available for this mode.'
    return
  }

  videoMarkerListEl.className = 'marker-list'
  videoMarkerListEl.innerHTML = items.map(item => `<div class="marker-item"><b>${escapeHtml(item.title)}</b> | Score: ${escapeHtml(formatNumber(item.score))}<div>${escapeHtml(item.note)}</div></div>`).join('')
}

function setVideoFrameStripMessage(text) {
  if(!videoFrameStripEl) return
  videoFrameStripEl.className = 'frame-strip empty-note'
  videoFrameStripEl.textContent = text
}

function extractFrameCaptureTimes(data, timeline) {
  const duration = getPreviewDurationSeconds()
  const candidates = []
  const segments = Array.isArray(data?.segments) ? data.segments : []

  segments
    .filter(seg => seg && (seg.start_time !== undefined || seg.start !== undefined))
    .sort((a, b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))
    .slice(0, 4)
    .forEach(seg => {
      const start = Number(seg.start_time ?? seg.start ?? 0)
      const end = Number(seg.end_time ?? seg.end ?? start)
      candidates.push((start + Math.max(start, end)) / 2)
    })

  ;[...timeline]
    .sort((a, b) => Number(b.y || 0) - Number(a.y || 0))
    .slice(0, 4)
    .forEach(point => candidates.push(Number(point.x)))

  if(candidates.length === 0) {
    ;[0.16, 0.38, 0.62, 0.84].forEach(ratio => candidates.push(duration * ratio))
  }

  const seen = new Set()
  return candidates
    .map(time => Math.max(0, Math.min(duration, Number(time) || 0)))
    .filter(time => {
      const key = time.toFixed(2)
      if(seen.has(key)) return false
      seen.add(key)
      return true
    })
    .slice(0, 4)
}

function nearestTimelineScore(timeline, targetX) {
  if(!timeline?.length) return null
  let closest = timeline[0]
  let minDistance = Math.abs(Number(closest.x) - targetX)
  timeline.forEach(point => {
    const distance = Math.abs(Number(point.x) - targetX)
    if(distance < minDistance) {
      closest = point
      minDistance = distance
    }
  })
  return closest?.y ?? null
}

async function seekVideoFrame(video, targetTime) {
  const maxTime = Math.max(0, (Number(video.duration) || 0) - 0.05)
  const safeTime = Math.max(0, Math.min(targetTime, maxTime))
  if(Math.abs(video.currentTime - safeTime) < 0.02) return

  await new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('error', onError)
    }
    const onSeeked = () => { cleanup(); resolve() }
    const onError = () => { cleanup(); reject(new Error('Failed to seek preview video.')) }
    video.addEventListener('seeked', onSeeked)
    video.addEventListener('error', onError)
    video.currentTime = safeTime
  })
}

async function renderVideoFrames(data, timeline) {
  if(!videoFrameStripEl) return

  const currentToken = ++frameRenderToken
  if(!previewUrl) {
    setVideoFrameStripMessage('Upload a video to preview key frames.')
    return
  }

  videoFrameStripEl.className = 'frame-strip'
  videoFrameStripEl.innerHTML = '<div class="empty-note">Generating key frame previews...</div>'

  const scratch = document.createElement('video')
  scratch.preload = 'auto'
  scratch.muted = true
  scratch.playsInline = true
  scratch.src = previewUrl
  scratch.load()

  try {
    if(scratch.readyState < 1) {
      await new Promise((resolve, reject) => {
        const cleanup = () => {
          scratch.removeEventListener('loadedmetadata', onReady)
          scratch.removeEventListener('error', onError)
        }
        const onReady = () => { cleanup(); resolve() }
        const onError = () => { cleanup(); reject(new Error('Preview metadata unavailable.')) }
        scratch.addEventListener('loadedmetadata', onReady)
        scratch.addEventListener('error', onError)
      })
    }

    const frameTimes = extractFrameCaptureTimes(data, timeline)
    const canvas = document.createElement('canvas')
    canvas.width = 320
    canvas.height = 180
    const ctx = canvas.getContext('2d')
    if(!ctx) {
      setVideoFrameStripMessage('Key frame preview is unavailable for this file.')
      return
    }
    const frames = []

    for(const time of frameTimes) {
      if(currentToken !== frameRenderToken) return
      await seekVideoFrame(scratch, time)
      ctx.drawImage(scratch, 0, 0, canvas.width, canvas.height)
      frames.push({
        image: canvas.toDataURL('image/jpeg', 0.82),
        time,
        score: nearestTimelineScore(timeline, time),
      })
    }

    if(currentToken !== frameRenderToken) return
    if(frames.length === 0) {
      setVideoFrameStripMessage('Key frame preview is unavailable for this file.')
      return
    }

    videoFrameStripEl.className = 'frame-strip'
    videoFrameStripEl.innerHTML = frames.map(frame => `
      <div class="frame-card">
        <img src="${frame.image}" alt="Video frame at ${escapeHtml(formatTime(frame.time))}">
        <div class="frame-caption">
          <b>${escapeHtml(formatTime(frame.time))}</b>${frame.score !== null ? ` | score ${escapeHtml(formatNumber(frame.score))}` : ''}
        </div>
      </div>
    `).join('')
  } catch {
    if(currentToken !== frameRenderToken) return
    setVideoFrameStripMessage('Key frame preview is unavailable for this file.')
  }
}

function buildAcousticProfile(data, timeline, records) {
  const metrics = data?.metrics && typeof data.metrics === 'object' ? data.metrics : {}
  const segments = Array.isArray(data?.segments) ? data.segments : []
  const scores = timeline.map(t => Number(t.y)).filter(v => !Number.isNaN(v))
  const durations = segments.map(s => Number(s.duration)).filter(v => !Number.isNaN(v) && v > 0)

  const scoreStd = metrics.score_std !== undefined ? clamp01(Number(metrics.score_std) * 2.4) : clamp01(std(scores) * 2.4)
  const durationCvRaw = metrics.duration_cv !== undefined ? Number(metrics.duration_cv) : (durations.length > 1 ? (std(durations) / Math.max(1e-6, avg(durations))) : 0)
  const durationCv = clamp01(durationCvRaw)

  const jumpRatio = metrics.jump_ratio !== undefined
    ? clamp01(Number(metrics.jump_ratio))
    : (() => {
        if(scores.length <= 1) return 0
        let jumps = 0
        for(let i = 1; i < scores.length; i += 1) if(Math.abs(scores[i] - scores[i - 1]) > 0.35) jumps += 1
        return clamp01(jumps / (scores.length - 1))
      })()

  const cvAudioScore = records.find(r => r.name === 'cv_audio' && r.score !== null && r.score !== undefined)
  const baseline = pickOverallRisk(data, records) ?? 0.5

  const profile = {
    labels: ['Pitch Variability', 'Speech Rhythm', 'Pause Pattern', 'Energy Stability', 'Prosody Shift', 'Arousal'],
    values: [
      scoreStd,
      clamp01(1 - durationCv * 0.75),
      durationCv,
      clamp01(1 - scoreStd * 0.8),
      jumpRatio,
      clamp01(cvAudioScore ? Number(cvAudioScore.score) : baseline)
    ]
  }

  const insights = []
  if(profile.values[0] > 0.55) insights.push('Speech variability elevated')
  if(profile.values[2] > 0.5) insights.push('Pause duration instability detected')
  if(profile.values[4] > 0.5) insights.push('Prosodic jumps across segments')
  if(profile.values[5] > 0.65) insights.push('High multimodal arousal')
  if(insights.length === 0) insights.push('Audio-acoustic pattern is relatively stable')

  return { profile, insights }
}

function renderAcousticRadar(profileData) {
  destroyChart('acousticRadar')
  const ctx = getChartContext(acousticRadarCanvas, 'acousticRadarChart')
  if(!ctx) return
  charts.acousticRadar = new Chart(ctx, {
    type: 'radar',
    data: { labels: profileData.labels, datasets: [{ label: 'Acoustic profile', data: profileData.values, borderColor: '#57d0c9', backgroundColor: 'rgba(87, 208, 201, 0.22)', pointBackgroundColor: '#57d0c9' }] },
    options: { maintainAspectRatio: false, scales: { r: { min: 0, max: 1, grid: { color: 'rgba(122, 165, 231, 0.2)' }, angleLines: { color: 'rgba(122, 165, 231, 0.2)' }, ticks: { color: '#9eb7dc', backdropColor: 'transparent' }, pointLabels: { color: '#bfd2f2' } } }, plugins: { legend: { display: false } } }
  })
}

function renderAcousticInsights(insights) {
  acousticInsightsEl.innerHTML = ''
  insights.forEach(text => {
    const chip = document.createElement('span')
    chip.className = 'insight-chip'
    chip.textContent = text
    acousticInsightsEl.appendChild(chip)
  })
}

function renderAcousticTimeline(timelineState) {
  destroyChart('acousticTimeline')
  const timeline = timelineState?.points || []
  if(!timeline || timeline.length === 0) {
    acousticTimelineHintEl.textContent = 'No timeline points for this mode.'
    return
  }

  const sorted = [...timeline]
    .map(p => ({ x: Number(p.x), y: Number(p.y) }))
    .filter(p => !Number.isNaN(p.x) && !Number.isNaN(p.y))
    .sort((a, b) => a.x - b.x)
  if(sorted.length === 0) {
    acousticTimelineHintEl.textContent = 'Timeline data is empty after filtering.'
    return
  }

  const compact = downsampleTimeline(sorted, 140)
  acousticTimelineHintEl.textContent = timelineState?.synthetic
    ? 'Showing a model-level fallback line because segment-level acoustic dynamics are unavailable.'
    : compact.length < sorted.length
      ? `Showing ${compact.length} of ${sorted.length} points for readability.`
      : `Showing ${compact.length} points.`
  const ctx = getChartContext(acousticTimelineCanvas, 'acousticTimelineChart')
  if(!ctx) return

  charts.acousticTimeline = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Acoustic risk dynamics',
        data: compact,
        parsing: false,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.2)',
        tension: 0.3,
        fill: true,
        pointRadius: 1.8
      }]
    },
    options: baseChartOptions({
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'linear',
          ticks: { color: '#9eb7dc' },
          grid: { color: 'rgba(122, 165, 231, 0.12)' },
          title: { display: true, text: 'time / index', color: '#9eb7dc' }
        },
        y: {
          beginAtZero: true,
          max: 1,
          ticks: { color: '#9eb7dc' },
          grid: { color: 'rgba(122, 165, 231, 0.12)' }
        }
      }
    })
  })
}

function collectSegments(...sources) {
  const pool = []
  sources.forEach(src => {
    if(Array.isArray(src?.segments)) pool.push(...src.segments)
  })

  const seen = new Set()
  return pool.filter(seg => {
    if(!seg || typeof seg !== 'object') return false
    const key = `${seg.segment_id || ''}|${seg.start_time ?? seg.start ?? ''}|${seg.end_time ?? seg.end ?? ''}|${seg.text || ''}`
    if(seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function buildTextPayload(data, rawPayload, overallScore) {
  const segments = collectSegments(
    data,
    rawPayload,
    rawPayload?.formatted_result,
    rawPayload?.formatted,
    rawPayload?.raw_result
  )

  const lines = []
  if(segments.length) segments.forEach(seg => { const text = String(seg?.text || '').trim(); if(text) lines.push(text) })
  if(lines.length === 0){
    const seenTexts = new Set()
    const textCandidates = [
      data?.raw_text,
      data?.transcript,
      data?.full_text,
      data?.text,
      data?.models?.nlp?.transcript,
      data?.models?.nlp?.full_text,
      rawPayload?.raw_text,
      rawPayload?.transcript,
      rawPayload?.full_text,
      rawPayload?.text,
      rawPayload?.models?.nlp?.transcript,
      rawPayload?.models?.nlp?.full_text,
      rawPayload?.raw_result?.raw_text,
      rawPayload?.raw_result?.transcript,
      rawPayload?.raw_result?.full_text,
      rawPayload?.raw_result?.text,
      rawPayload?.formatted_result?.raw_text,
      rawPayload?.formatted_result?.transcript,
      rawPayload?.formatted_result?.full_text,
      rawPayload?.formatted_result?.text
    ]

    textCandidates.forEach(candidate => {
      if(typeof candidate === 'string' && candidate.trim()){
        const clean = candidate.trim()
        if(seenTexts.has(clean)) return
        seenTexts.add(clean)
        lines.push(clean)
      }
    })
  }

  const text = lines.join(' ').trim()
  const stopwords = new Set(['this', 'that', 'with', 'from', 'have', 'were', 'there', 'about', 'which', 'would', 'could', 'should', 'their', 'them', 'what', 'when', 'where', 'while', 'into', 'your', 'just', 'been', 'very', 'more', 'also', 'than', 'then', 'they', 'because', 'after', 'before', 'still', 'some', 'much', 'such', 'like', 'feel', 'feels', 'feeling'])

  const termMap = new Map()
  segments.forEach(seg => {
    const segText = String(seg?.text || '').toLowerCase()
    if(!segText) return
    const score = seg?.risk_score !== undefined ? Number(seg.risk_score) : (overallScore ?? 0.5)
    const tokens = segText.match(/[\p{L}']{4,}/gu) || []
    tokens.forEach(token => {
      if(stopwords.has(token)) return
      if(!termMap.has(token)) termMap.set(token, { term: token, count: 0, maxRisk: 0 })
      const current = termMap.get(token)
      current.count += 1
      current.maxRisk = Math.max(current.maxRisk, Number.isNaN(score) ? 0 : score)
    })
  })

  const ranked = Array.from(termMap.values()).sort((a, b) => (b.maxRisk * 2 + b.count * 0.08) - (a.maxRisk * 2 + a.count * 0.08)).slice(0, 10).map(item => ({ ...item, weight: clamp01(item.maxRisk * 0.7 + Math.min(1, item.count / 6) * 0.3) }))
  return { text, terms: ranked }
}

function renderTranscript(payload) {
  activeTermMap = new Map()
  payload.terms.forEach(t => activeTermMap.set(t.term.toLowerCase(), t))

  if(!payload.text) {
    transcriptTextEl.className = 'transcript-box empty-note'
    transcriptTextEl.textContent = 'No transcript content available for this mode.'
    return
  }

  const clippedText = payload.text.length > 6000 ? `${payload.text.slice(0, 6000)} ...` : payload.text
  let html = escapeHtml(clippedText)
  if(payload.terms.length) {
    const pattern = new RegExp(`\\b(${payload.terms.map(t => escapeRegex(t.term)).join('|')})\\b`, 'gi')
    html = html.replace(pattern, match => `<span class="term-hit" data-term="${match.toLowerCase()}">${match}</span>`)
  }

  transcriptTextEl.className = 'transcript-box'
  transcriptTextEl.innerHTML = html
}

function setActiveTranscriptTerm(termLower) {
  transcriptTextEl.querySelectorAll('.term-hit').forEach(node => node.classList.toggle('is-active', node.dataset.term === termLower))
}

function renderTextMarkerInfo(termData, overallScore) {
  if(!termData) {
    textMarkerInfoEl.innerHTML = `<div><b>Word:</b> -</div><div><b>Risk score:</b> ${escapeHtml(formatNumber(overallScore))}</div><div><b>Contribution weight:</b> -</div><div class="empty-note" style="margin-top:6px;">Click highlighted text markers to inspect influence.</div>`
    return
  }
  textMarkerInfoEl.innerHTML = `<div><b>Word:</b> ${escapeHtml(termData.term)}</div><div><b>Risk score:</b> ${escapeHtml(formatNumber(termData.maxRisk))}</div><div><b>Contribution weight:</b> ${escapeHtml(formatNumber(termData.weight))}</div><div><b>Count:</b> ${escapeHtml(termData.count)}</div>`
}

function prettyFeatureName(name) {
  return String(name || '').replaceAll('_', ' ').replace(/\bsma3nz\b/gi, '').replace(/\bamean\b/gi, 'avg').replace(/\s+/g, ' ').trim()
}

function extractInfluenceRows(data) {
  const segments = Array.isArray(data?.segments) ? data.segments : []
  const map = new Map()
  segments.forEach(seg => {
    const top = Array.isArray(seg?.top_features) ? seg.top_features : []
    top.forEach(feat => {
      const key = feat.feature || feat.type || 'unknown_feature'
      if(!map.has(key)) map.set(key, { feature: key, type: feat.type || 'unknown', contribution: 0, count: 0, valueSum: 0 })
      const rec = map.get(key)
      rec.contribution += Number(feat.contribution || 0)
      rec.valueSum += Number(feat.value || 0)
      rec.count += 1
    })
  })
  return Array.from(map.values()).map(row => ({ ...row, contribution: row.count ? row.contribution / row.count : 0, valueMean: row.count ? row.valueSum / row.count : 0 })).sort((a, b) => b.contribution - a.contribution).slice(0, 10)
}

function renderInfluenceTable(data) {
  const rows = extractInfluenceRows(data)
  if(rows.length === 0) {
    influenceTableWrapEl.innerHTML = '<div class="empty-note">No feature contribution data available.</div>'
    return
  }
  let html = '<table class="model-table"><thead><tr><th>Feature</th><th>Type</th><th>Weight</th><th>Mean Value</th></tr></thead><tbody>'
  rows.forEach(row => {
    html += `<tr><td>${escapeHtml(prettyFeatureName(row.feature))}</td><td>${escapeHtml(row.type)}</td><td>${escapeHtml(formatNumber(row.contribution))}</td><td>${escapeHtml(formatNumber(row.valueMean))}</td></tr>`
  })
  influenceTableWrapEl.innerHTML = `${html}</tbody></table>`
}

function renderControlChart(records, data) {
  if(!controlCanvas) return
  destroyChart('control')
  const subjectScore = clamp01(pickOverallRisk(data, records) ?? 0)
  const nonEnsemble = records.filter(r => r.name !== 'ensemble' && r.score !== null && r.score !== undefined).map(r => Number(r.score)).filter(v => !Number.isNaN(v))
  const controlScore = nonEnsemble.length ? clamp01(avg(nonEnsemble)) : clamp01(Math.max(0.25, 1 - subjectScore * 0.8))
  const ctx = getChartContext(controlCanvas, 'controlChart')
  if(!ctx) return

  charts.control = new Chart(ctx, {
    type: 'bar',
    data: { labels: ['Subject', 'Control'], datasets: [{ label: 'Subject profile', data: [subjectScore, null], backgroundColor: '#4a83ff' }, { label: 'Anonymized control', data: [null, controlScore], backgroundColor: '#f2b24a' }] },
    options: baseChartOptions({ plugins: { legend: { position: 'top', labels: { color: '#bfd2f2' } } } })
  })
}
function renderAll(data, modeCfg, rawPayload) {
  const records = extractModelRecords(data, modeCfg)
  const timeline = extractTimeline(data)
  const score = pickOverallRisk(data, records)
  const level = pickOverallLevel(data, score)
  const videoTimeline = resolveTimeline(timeline, pickRecordScore(records, ['cv', 'video', 'ensemble']) ?? score)
  const acousticTimeline = resolveTimeline(timeline, pickRecordScore(records, ['cv_audio', 'nlp', 'ensemble']) ?? score)

  renderGauge(score)
  setRiskBadge(level)
  renderSummary(data, modeCfg, records, score, level)
  renderScoreChart(records)
  renderStatusChart(records)
  renderDistributionChart(data, records)
  renderVideoTimelineChart(videoTimeline)
  renderVideoMarkers(data, videoTimeline.points, records)
  void renderVideoFrames(data, videoTimeline.points)

  const acoustic = buildAcousticProfile(data, acousticTimeline.points, records)
  renderAcousticRadar(acoustic.profile)
  renderAcousticInsights(acoustic.insights)
  renderAcousticTimeline(acousticTimeline)

  const textPayload = buildTextPayload(data, rawPayload, score)
  renderTranscript(textPayload)
  renderInfluenceTable(data)

  if(textPayload.terms.length) {
    renderTextMarkerInfo(textPayload.terms[0], score)
    setActiveTranscriptTerm(textPayload.terms[0].term.toLowerCase())
  } else {
    renderTextMarkerInfo(null, score)
    setActiveTranscriptTerm(null)
  }

  rawJsonEl.textContent = JSON.stringify(data, null, 2)
}

function clearUI() {
  summaryCardEl.innerHTML = '<h3>Summary</h3><div class="empty-note">No data yet.</div>'
  rawJsonEl.textContent = '{}'

  riskScoreValueEl.textContent = '-'
  setRiskBadge('unknown')

  transcriptTextEl.className = 'transcript-box empty-note'
  transcriptTextEl.textContent = 'Run analysis to populate transcript markers.'
  textMarkerInfoEl.innerHTML = '<div class="empty-note">No marker selected.</div>'
  influenceTableWrapEl.innerHTML = '<div class="empty-note">No feature contribution data available.</div>'

  videoMarkerListEl.className = 'marker-list empty-note'
  videoMarkerListEl.textContent = 'Run analysis to see marker signals.'
  setVideoFrameStripMessage('Run analysis to generate key frame previews.')
  if(videoTimelineHintEl) videoTimelineHintEl.textContent = 'Awaiting CV markers.'
  acousticInsightsEl.innerHTML = '<span class="insight-chip">Awaiting audio-acoustic features</span>'
  acousticTimelineHintEl.textContent = 'Run analysis to populate the acoustic timeline.'

  clearCharts()
  resetPipelineState()
  setStatus('Ready.')
}

async function sendRequest(modeCfg) {
  let body = null
  if(modeCfg.needsFile) {
    if(!videoInput.files || videoInput.files.length === 0) {
      setStatus('Select a video file for the selected mode.', 'error')
      return
    }
    body = new FormData()
    body.append('file', videoInput.files[0])
  }

  const url = apiBase + modeCfg.endpoint
  startPipelineAnimation()
  setStatus(`Request: ${modeCfg.method} ${modeCfg.endpoint}`)

  const startedAt = performance.now()
  try {
    const res = await fetch(url, { method: modeCfg.method, body })
    const text = await res.text()
    const elapsedMs = Math.round(performance.now() - startedAt)
    const sizeBytes = new TextEncoder().encode(text).length

    if(!res.ok) {
      finishPipeline(false)
      setStatus(`Error ${res.status}: ${res.statusText}`, 'error')
      summaryCardEl.innerHTML = `<h3>Summary</h3><div class="empty-note">${escapeHtml(text)}</div>`
      rawJsonEl.textContent = text
      clearCharts()
      return
    }

    let json = {}
    try { json = JSON.parse(text) } catch { json = { success: true, raw_text: text } }

    const data = normalizeResponse(json)
    try {
      renderAll(data, modeCfg, json)
    } catch (err) {
      console.error('Render error', err)
      finishPipeline(false)
      setStatus(`Render error: ${err.message}`, 'error')
      summaryCardEl.innerHTML = `<h3>Summary</h3><div class="empty-note">${escapeHtml(err.message)}</div>`
      rawJsonEl.textContent = JSON.stringify({ error: err.message, data }, null, 2)
      clearCharts()
      return
    }
    finishPipeline(true)
    setStatus(`Done: ${modeCfg.label} (${elapsedMs} ms)`, 'ok')
  } catch (err) {
    finishPipeline(false)
    setStatus(`Fetch error: ${err.message}`, 'error')
    summaryCardEl.innerHTML = `<h3>Summary</h3><div class="empty-note">${escapeHtml(err.message)}</div>`
    rawJsonEl.textContent = JSON.stringify({ error: err.message }, null, 2)
    clearCharts()
  }
}

runBtn.addEventListener('click', async () => {
  const modeCfg = getSelectedMode()
  await sendRequest(modeCfg)
})

clearBtn.addEventListener('click', () => { clearUI() })

toggleRawBtn.addEventListener('click', () => {
  rawPanelEl.classList.toggle('hidden')
  toggleRawBtn.textContent = rawPanelEl.classList.contains('hidden') ? 'View Raw JSON' : 'Hide Raw JSON'
})

jumpAnalyzeBtn.addEventListener('click', () => {
  document.getElementById('analysis-input')?.scrollIntoView({ behavior: 'smooth' })
})

videoInput.addEventListener('change', () => {
  const file = videoInput.files?.[0]
  if(!file) {
    selectedFileNameEl.textContent = 'No file selected'
    durationHintEl.textContent = 'Awaiting upload'
    setVideoFrameStripMessage('Upload a video to preview key frames.')
    if(videoTimelineHintEl) videoTimelineHintEl.textContent = 'Awaiting CV markers.'
    previewVideo.removeAttribute('src')
    previewVideo.load()
    return
  }

  selectedFileNameEl.textContent = file.name
  durationHintEl.textContent = `${Math.round(file.size / 1024 / 1024 * 10) / 10} MB`
  setVideoFrameStripMessage('Run analysis to generate key frame previews.')

  if(previewUrl) URL.revokeObjectURL(previewUrl)
  previewUrl = URL.createObjectURL(file)
  previewVideo.src = previewUrl
})

previewVideo.addEventListener('loadedmetadata', () => {
  const sizeMb = videoInput.files?.[0] ? `${Math.round(videoInput.files[0].size / 1024 / 1024 * 10) / 10} MB` : ''
  const duration = Number(previewVideo.duration)
  const durationLabel = Number.isFinite(duration) && duration > 0 ? formatTime(duration) : 'Unknown duration'
  durationHintEl.textContent = sizeMb ? `${durationLabel} | ${sizeMb}` : durationLabel
})

transcriptTextEl.addEventListener('click', event => {
  const target = event.target
  if(!(target instanceof HTMLElement) || !target.classList.contains('term-hit')) return
  const term = (target.dataset.term || '').toLowerCase()
  const termData = activeTermMap.get(term)
  if(!termData) return
  renderTextMarkerInfo(termData, null)
  setActiveTranscriptTerm(term)
})

buildModeOptions()
clearUI()
