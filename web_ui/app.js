const apiBase = window.apiBase || 'http://localhost:8000'

const modeOptionsEl = document.getElementById('modeOptions')
const videoInput = document.getElementById('videoInput')
const runBtn = document.getElementById('runBtn')
const clearBtn = document.getElementById('clearBtn')
const statusEl = document.getElementById('status')
const summaryCardEl = document.getElementById('summaryCard')
const metaCardEl = document.getElementById('metaCard')
const rawJsonEl = document.getElementById('rawJson')

const scoreCanvas = document.getElementById('scoreChart')
const statusCanvas = document.getElementById('statusChart')
const timelineCanvas = document.getElementById('timelineChart')
const distributionCanvas = document.getElementById('distributionChart')

const MODE_CONFIGS = [
  {
    id: 'all',
    label: 'Все модели (NLP + CV + CV+Audio)',
    endpoint: '/predict?include_cv=true&include_cv_audio=true&video_sample_every=8&cv_audio_sample_rate=0.25',
    method: 'POST',
    needsFile: true
  },
  { id: 'nlp', label: 'Только NLP', endpoint: '/test/nlp', method: 'POST', needsFile: true },
  { id: 'cv', label: 'Только CV', endpoint: '/test/cv?sample_every=8', method: 'POST', needsFile: true },
  { id: 'cv_audio', label: 'Только CV+Audio', endpoint: '/predict/cv-audio?sample_rate=0.25', method: 'POST', needsFile: true },
  { id: 'deception', label: '4. Deception detection', endpoint: '/test/deception', method: 'POST', needsFile: true },
  { id: 'emotion_av', label: '5. Emotion audio+video', endpoint: '/test/emotion-av?sample_every=8&sample_rate=0.25', method: 'POST', needsFile: true },
  { id: 'anomaly_audio', label: '6. Anomaly audio', endpoint: '/test/anomaly/audio', method: 'POST', needsFile: true },
  { id: 'anomaly_video', label: '6. Anomaly video', endpoint: '/test/anomaly/video?sample_every=8', method: 'POST', needsFile: true },
  { id: 'anomaly_text', label: '6. Anomaly text', endpoint: '/test/anomaly/text', method: 'POST', needsFile: true },
  // { id: 'health', label: 'Health', endpoint: '/health', method: 'GET', needsFile: false }
]

const charts = {
  score: null,
  status: null,
  timeline: null,
  distribution: null
}

function setStatus(text, state){
  statusEl.textContent = text
  if(state){
    statusEl.dataset.state = state
  }else{
    delete statusEl.dataset.state
  }
}

function escapeHtml(str){
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function formatNumber(v){
  if(v === null || v === undefined) return '-'
  const n = Number(v)
  if(Number.isNaN(n)) return String(v)
  return (Math.round(n * 1000) / 1000).toString()
}

function modelTitle(name){
  const map = {
    nlp: 'NLP',
    cv: 'CV',
    video: 'CV',
    cv_audio: 'CV+Audio',
    deception: 'Deception',
    emotion_av: 'Emotion AV',
    anomaly_text: 'Anomaly text',
    anomaly_audio: 'Anomaly audio',
    anomaly_video: 'Anomaly video',
    health: 'Health'
  }
  return map[name] || name
}

function pickScore(obj){
  if(!obj || typeof obj !== 'object') return null
  const keys = ['probability', 'risk_score', 'deception_score', 'emotion_score', 'anomaly_score', 'ensemble_score']
  for(const key of keys){
    if(obj[key] !== undefined && obj[key] !== null && !Number.isNaN(Number(obj[key]))){
      return Number(obj[key])
    }
  }
  return null
}

function normalizeResponse(json){
  if(json && typeof json === 'object'){
    if(json.formatted_result) return json.formatted_result
    if(json.formatted) return json.formatted
  }
  return json
}

function buildModeOptions(){
  modeOptionsEl.innerHTML = ''
  MODE_CONFIGS.forEach((cfg, index) => {
    const label = document.createElement('label')
    label.innerHTML = `<input type="radio" name="mode" value="${cfg.id}" ${index === 0 ? 'checked' : ''}> ${cfg.label}`
    modeOptionsEl.appendChild(label)
  })
}

function getSelectedMode(){
  const selected = document.querySelector('input[name="mode"]:checked')
  return MODE_CONFIGS.find(m => m.id === selected?.value) || MODE_CONFIGS[0]
}

function extractModelRecords(data, modeCfg){
  const records = []

  if(data && data.models && typeof data.models === 'object'){
    Object.entries(data.models).forEach(([name, value]) => {
      if(!value || typeof value !== 'object') return
      records.push({
        name,
        success: value.success,
        prediction: value.prediction,
        prediction_label: value.prediction_label,
        risk_level: value.risk_level,
        score: pickScore(value),
        error: value.error
      })
    })
  }

  if(data && data.overall && typeof data.overall === 'object'){
    const overall = data.overall
    if(overall.ensemble && overall.ensemble.ensemble_score !== undefined){
      records.push({
        name: 'ensemble',
        success: true,
        prediction: overall.ensemble.ensemble_prediction,
        prediction_label: overall.ensemble.ensemble_prediction_label,
        risk_level: overall.ensemble.risk_level,
        score: Number(overall.ensemble.ensemble_score),
        error: null
      })
    }
  }

  if(records.length === 0){
    records.push({
      name: modeCfg.id,
      success: data?.success,
      prediction: data?.prediction,
      prediction_label: data?.prediction_label,
      risk_level: data?.risk_level,
      score: pickScore(data),
      error: data?.error
    })

    if(data?.cv_audio_probability !== undefined){
      records.push({
        name: 'cv_audio',
        success: data?.cv_audio_success,
        prediction: null,
        prediction_label: null,
        risk_level: null,
        score: Number(data.cv_audio_probability),
        error: data?.cv_audio_error
      })
    }
  }

  const dedup = new Map()
  records.forEach(r => {
    if(!dedup.has(r.name)) dedup.set(r.name, r)
  })

  return Array.from(dedup.values())
}

function extractTimeline(data){
  if(Array.isArray(data?.time_series)){
    return data.time_series
      .map((x, idx) => ({
        x: x.start !== undefined ? Number(x.start) : idx,
        y: x.score !== undefined ? Number(x.score) : null
      }))
      .filter(p => p.y !== null && !Number.isNaN(p.y))
  }

  if(Array.isArray(data?.segments)){
    return data.segments
      .map((seg, idx) => ({
        x: seg.start_time !== undefined ? Number(seg.start_time) : idx,
        y: seg.risk_score !== undefined ? Number(seg.risk_score) : null
      }))
      .filter(p => p.y !== null && !Number.isNaN(p.y))
  }

  return []
}

function renderSummary(data, modeCfg, records){
  const rows = []
  const overall = data?.overall || data || {}

  rows.push(['Режим', modeCfg.label])
  rows.push(['Успешно', overall.success])
  if(overall.prediction !== undefined) rows.push(['Prediction', overall.prediction])
  if(overall.prediction_label !== undefined) rows.push(['Prediction label', overall.prediction_label])
  if(overall.risk_level !== undefined) rows.push(['Risk level', overall.risk_level])

  const score = pickScore(overall)
  if(score !== null) rows.push(['Score', formatNumber(score)])

  if(data?.modality !== undefined) rows.push(['Modality', data.modality])
  if(data?.segments_used !== undefined) rows.push(['Segments used', data.segments_used])

  if(overall.ensemble?.models_used){
    rows.push(['Ensemble models', overall.ensemble.models_used.join(', ')])
  }

  let html = '<h3>Summary</h3><table class="kv">'
  rows.forEach(([k, v]) => {
    html += `<tr><td><b>${escapeHtml(k)}</b></td><td>${escapeHtml(v ?? '-')}</td></tr>`
  })
  html += '</table>'

  html += '<h3 style="margin-top:10px;">Model metrics</h3>'
  html += '<table class="model-table">'
  html += '<thead><tr><th>Model</th><th>Success</th><th>Pred</th><th>Risk</th><th>Score</th><th>Error</th></tr></thead><tbody>'
  records.forEach(r => {
    const successClass = r.success === true ? 'model-ok' : (r.success === false ? 'model-bad' : '')
    html += `<tr>
      <td>${escapeHtml(modelTitle(r.name))}</td>
      <td class="${successClass}">${escapeHtml(r.success ?? '-')}</td>
      <td>${escapeHtml(r.prediction ?? '-')}</td>
      <td>${escapeHtml(r.risk_level ?? '-')}</td>
      <td>${escapeHtml(formatNumber(r.score))}</td>
      <td>${escapeHtml(r.error ?? '-')}</td>
    </tr>`
  })
  html += '</tbody></table>'

  summaryCardEl.innerHTML = html
}

function renderMeta(meta, data){
  const keys = data && typeof data === 'object' ? Object.keys(data) : []
  const html = `
    <h3>Request meta</h3>
    <table class="kv">
      <tr><td><b>Endpoint</b></td><td>${escapeHtml(meta.endpoint)}</td></tr>
      <tr><td><b>Method</b></td><td>${escapeHtml(meta.method)}</td></tr>
      <tr><td><b>Elapsed</b></td><td>${escapeHtml(meta.elapsedMs + ' ms')}</td></tr>
      <tr><td><b>Response size</b></td><td>${escapeHtml(meta.sizeBytes + ' bytes')}</td></tr>
      <tr><td><b>Root keys</b></td><td>${escapeHtml(keys.join(', ') || '-')}</td></tr>
    </table>
  `
  metaCardEl.innerHTML = html
}

function clearCharts(){
  Object.keys(charts).forEach(k => {
    if(charts[k]){
      charts[k].destroy()
      charts[k] = null
    }
  })
}

function renderScoreChart(records){
  const items = records.filter(r => r.score !== null && r.score !== undefined && !Number.isNaN(Number(r.score)))
  if(items.length === 0) return

  charts.score = new Chart(scoreCanvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: items.map(i => modelTitle(i.name)),
      datasets: [{
        label: 'Score / Probability',
        data: items.map(i => Number(i.score)),
        backgroundColor: ['#1f6feb', '#0ea5e9', '#16a34a', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6']
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  })
}

function renderStatusChart(records){
  const counts = { ok: 0, fail: 0, unknown: 0 }
  records.forEach(r => {
    if(r.success === true) counts.ok += 1
    else if(r.success === false) counts.fail += 1
    else counts.unknown += 1
  })

  const values = [counts.ok, counts.fail, counts.unknown]
  if(values.reduce((a, b) => a + b, 0) === 0) return

  charts.status = new Chart(statusCanvas.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['OK', 'Fail', 'Unknown'],
      datasets: [{ data: values, backgroundColor: ['#16a34a', '#dc2626', '#94a3b8'] }]
    },
    options: { plugins: { legend: { position: 'bottom' } } }
  })
}

function renderTimelineChart(timeline){
  if(!timeline || timeline.length === 0) return

  const sorted = timeline
    .map(p => ({ x: Number(p.x), y: Number(p.y) }))
    .filter(p => !Number.isNaN(p.x) && !Number.isNaN(p.y))
    .sort((a, b) => a.x - b.x)

  if(sorted.length === 0) return

  charts.timeline = new Chart(timelineCanvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: sorted.map(p => p.x.toFixed(1)),
      datasets: [{
        label: 'Score over time',
        data: sorted.map(p => p.y),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.2)',
        tension: 0.25,
        fill: true
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 1 },
        x: { title: { display: true, text: 'time / index' } }
      }
    }
  })
}

function renderDistributionChart(data){
  if(data?.emotion_distribution && Object.keys(data.emotion_distribution).length > 0){
    const entries = Object.entries(data.emotion_distribution)
    charts.distribution = new Chart(distributionCanvas.getContext('2d'), {
      type: 'pie',
      data: {
        labels: entries.map(([k]) => k),
        datasets: [{ data: entries.map(([, v]) => Number(v)), backgroundColor: ['#1f6feb', '#16a34a', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'] }]
      },
      options: { plugins: { legend: { position: 'bottom' } } }
    })
    return
  }

  if(data?.indicators && Object.keys(data.indicators).length > 0){
    const entries = Object.entries(data.indicators)
    charts.distribution = new Chart(distributionCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: entries.map(([k]) => k),
        datasets: [{ data: entries.map(([, v]) => Number(v)), backgroundColor: '#8b5cf6' }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, max: 1 } }
      }
    })
    return
  }

  if(data?.components && Object.keys(data.components).length > 0){
    const entries = Object.entries(data.components)
    charts.distribution = new Chart(distributionCanvas.getContext('2d'), {
      type: 'radar',
      data: {
        labels: entries.map(([k]) => k),
        datasets: [{
          label: 'Components',
          data: entries.map(([, v]) => Number(v)),
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.2)'
        }]
      },
      options: { scales: { r: { beginAtZero: true, max: 1 } } }
    })
    return
  }

  if(data?.overall?.ensemble?.individual_scores){
    const entries = Object.entries(data.overall.ensemble.individual_scores)
    charts.distribution = new Chart(distributionCanvas.getContext('2d'), {
      type: 'polarArea',
      data: {
        labels: entries.map(([k]) => modelTitle(k)),
        datasets: [{ data: entries.map(([, v]) => Number(v)), backgroundColor: ['#1f6feb', '#16a34a', '#f59e0b', '#ef4444', '#8b5cf6'] }]
      },
      options: { plugins: { legend: { position: 'bottom' } } }
    })
  }
}

function renderAll(data, modeCfg, meta){
  const records = extractModelRecords(data, modeCfg)
  const timeline = extractTimeline(data)

  renderSummary(data, modeCfg, records)
  renderMeta(meta, data)

  clearCharts()
  renderScoreChart(records)
  renderStatusChart(records)
  renderTimelineChart(timeline)
  renderDistributionChart(data)

  rawJsonEl.textContent = JSON.stringify(data, null, 2)
}

function clearUI(){
  summaryCardEl.innerHTML = '<h3>Summary</h3><div class="empty-note">Нет данных.</div>'
  metaCardEl.innerHTML = '<h3>Request meta</h3><div class="empty-note">Нет данных.</div>'
  rawJsonEl.textContent = '{}'
  clearCharts()
  setStatus('Готово.')
}

async function sendRequest(modeCfg){
  let body = null
  if(modeCfg.needsFile){
    if(!videoInput.files || videoInput.files.length === 0){
      setStatus('Выберите файл для выбранного режима.', 'error')
      return
    }
    body = new FormData()
    body.append('file', videoInput.files[0])
  }

  const url = apiBase + modeCfg.endpoint
  setStatus(`Запрос: ${modeCfg.method} ${modeCfg.endpoint}`)

  const startedAt = performance.now()
  try{
    const res = await fetch(url, {
      method: modeCfg.method,
      body
    })

    const text = await res.text()
    const elapsedMs = Math.round(performance.now() - startedAt)
    const sizeBytes = new TextEncoder().encode(text).length

    if(!res.ok){
      setStatus(`Ошибка ${res.status}: ${res.statusText}`, 'error')
      summaryCardEl.innerHTML = `<h3>Summary</h3><div class="empty-note">${escapeHtml(text)}</div>`
      metaCardEl.innerHTML = `<h3>Request meta</h3><div class="empty-note">Endpoint: ${escapeHtml(modeCfg.endpoint)}; ${res.status}</div>`
      rawJsonEl.textContent = text
      clearCharts()
      return
    }

    let json = {}
    try {
      json = JSON.parse(text)
    } catch {
      json = { success: true, raw_text: text }
    }

    const data = normalizeResponse(json)
    renderAll(data, modeCfg, {
      endpoint: modeCfg.endpoint,
      method: modeCfg.method,
      elapsedMs,
      sizeBytes
    })

    setStatus(`Готово: ${modeCfg.label} (${elapsedMs} ms)`, 'ok')
  } catch (err) {
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

clearBtn.addEventListener('click', () => {
  clearUI()
})

buildModeOptions()
clearUI()
