const apiBase = window.apiBase || 'http://localhost:8000'

const videoInput = document.getElementById('videoInput')
const runBtn = document.getElementById('runBtn')
const clearBtn = document.getElementById('clearBtn')
const status = document.getElementById('status')
const textResult = document.getElementById('textResult')
const probChartEl = document.getElementById('probChart')

console.log('[app.js] Using apiBase:', apiBase)

let chart = null

function setStatus(t){ status.textContent = t }

function clearResults(){
  textResult.innerHTML = ''
  setStatus('Готово.')
  if(chart){ chart.destroy(); chart = null }
}

function prettyNumber(n){
  if(n === null || n === undefined || Number.isNaN(Number(n))) return '-'
  return (Math.round(Number(n) * 1000) / 1000).toString()
}

function prettyModelName(name){
  const map = {
    nlp: 'NLP модель',
    cv: 'CV модель',
    video: 'CV модель',
    cv_audio: 'CV+Audio модель',
    deception: 'Deception detection',
    emotion_av: 'Emotion audio+video',
    anomaly_text: 'Anomaly text',
    anomaly_audio: 'Anomaly audio',
    anomaly_video: 'Anomaly video'
  }
  return map[name] || name
}

function scoreFromObject(obj){
  if(!obj || typeof obj !== 'object') return null
  const keys = ['probability', 'risk_score', 'deception_score', 'emotion_score', 'anomaly_score', 'ensemble_score']
  for(const k of keys){
    if(obj[k] !== undefined && obj[k] !== null && !Number.isNaN(Number(obj[k]))){
      return Number(obj[k])
    }
  }
  return null
}

function escapeHtml(str){
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function renderStructured(data){
  const overall = data.overall || {}
  const models = data.models || {}
  const lines = []

  lines.push('<b>Общий результат</b>')
  lines.push(`Успешно: ${overall.success}`)
  if(overall.prediction !== undefined) lines.push(`Предсказание: ${overall.prediction}`)
  if(overall.risk_level !== undefined) lines.push(`Уровень риска: ${overall.risk_level}`)
  if(overall.risk_score !== undefined) lines.push(`Риск-скор: ${prettyNumber(overall.risk_score)}`)
  if(overall.ensemble && overall.ensemble.ensemble_score !== undefined){
    lines.push(`Ensemble score: ${prettyNumber(overall.ensemble.ensemble_score)}`)
  }

  lines.push('<hr><b>Модели</b>')
  const modelEntries = Object.entries(models)
  if(modelEntries.length === 0){
    lines.push('Модельные данные отсутствуют')
  }

  for(const [name, m] of modelEntries){
    if(!m) continue
    lines.push(`<b>${prettyModelName(name)}</b>`)
    if(m.success !== undefined) lines.push(`Успешно: ${m.success}`)
    if(m.prediction !== undefined) lines.push(`Предсказание: ${m.prediction}`)
    if(m.prediction_label !== undefined) lines.push(`Метка: ${m.prediction_label}`)
    if(m.risk_level !== undefined) lines.push(`Уровень риска: ${m.risk_level}`)
    const score = scoreFromObject(m)
    if(score !== null) lines.push(`Скор: ${prettyNumber(score)}`)
    if(m.error) lines.push(`Ошибка: ${m.error}`)
  }

  if(data.error){
    lines.push('<hr><b>Ошибка</b>')
    lines.push(escapeHtml(data.error))
  }

  textResult.innerHTML = lines.map(l => `<div>${l}</div>`).join('')
}

function renderGeneric(data){
  const lines = []
  lines.push('<b>Ответ модели</b>')

  const preferredOrder = [
    'success',
    'prediction',
    'prediction_label',
    'risk_level',
    'probability',
    'risk_score',
    'deception_score',
    'emotion_score',
    'anomaly_score',
    'error'
  ]

  for(const key of preferredOrder){
    if(data[key] !== undefined){
      const v = typeof data[key] === 'object' ? JSON.stringify(data[key], null, 2) : data[key]
      lines.push(`<b>${key}</b>: ${escapeHtml(v)}`)
    }
  }

  lines.push('<hr><b>Полный JSON</b>')
  lines.push(`<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`)
  textResult.innerHTML = lines.map(l => `<div>${l}</div>`).join('')
}

function collectChartSeries(data){
  const collected = []

  if(data.overall && data.overall.ensemble && data.overall.ensemble.individual_scores){
    for(const [name, score] of Object.entries(data.overall.ensemble.individual_scores)){
      if(score !== null && score !== undefined && !Number.isNaN(Number(score))){
        collected.push({ label: prettyModelName(name), value: Number(score) })
      }
    }
    if(data.overall.ensemble.ensemble_score !== undefined && data.overall.ensemble.ensemble_score !== null){
      collected.push({ label: 'Ensemble', value: Number(data.overall.ensemble.ensemble_score) })
    }
  }

  if(data.models){
    for(const [name, model] of Object.entries(data.models)){
      const score = scoreFromObject(model)
      if(score !== null){
        collected.push({ label: prettyModelName(name), value: score })
      }
    }
  }

  const topLevelScore = scoreFromObject(data)
  if(topLevelScore !== null){
    const label = data.modality ? `${data.modality} score` : 'Score'
    collected.push({ label, value: topLevelScore })
  }

  const uniq = new Map()
  for(const item of collected){
    if(!uniq.has(item.label)) uniq.set(item.label, item.value)
  }

  return {
    labels: Array.from(uniq.keys()),
    values: Array.from(uniq.values())
  }
}

function renderChart(data){
  const { labels, values } = collectChartSeries(data)
  if(chart){ chart.destroy(); chart = null }
  if(labels.length === 0) return

  const colors = [
    '#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#f97316', '#84cc16', '#06b6d4'
  ]

  chart = new Chart(probChartEl.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Score / Probability',
        data: values,
        backgroundColor: labels.map((_, i) => colors[i % colors.length])
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 1 }
      }
    }
  })
}

function normalizeResponse(json){
  if(json.formatted_result) return json.formatted_result
  if(json.formatted) return json.formatted
  return json
}

function renderResult(data){
  if(data && (data.overall || data.models)){
    renderStructured(data)
  }else{
    renderGeneric(data)
  }
  renderChart(data)
}

async function sendFileTo(endpoint){
  if(!videoInput.files || videoInput.files.length === 0){
    setStatus('Выберите файл')
    return
  }

  const file = videoInput.files[0]
  const fd = new FormData()
  fd.append('file', file)

  setStatus('Отправка на ' + apiBase + endpoint + '...')
  console.log('[sendFileTo] POST', apiBase + endpoint, 'file:', file.name)

  try{
    const res = await fetch(apiBase + endpoint, { method: 'POST', body: fd })
    console.log('[sendFileTo] Response status:', res.status)

    if(!res.ok){
      const txt = await res.text()
      console.error('[sendFileTo] Error response:', txt)
      setStatus('Ошибка от сервера: ' + res.status)
      textResult.textContent = txt
      return
    }

    const data = await res.json()
    const normalized = normalizeResponse(data)
    renderResult(normalized)
    setStatus('Готово')
  }catch(err){
    console.error('[sendFileTo] Fetch error:', err)
    setStatus('Ошибка: ' + err.message)
    textResult.textContent = 'Fetch error: ' + err.message + '\n\nПроверьте консоль браузера (F12).'
  }
}

runBtn.addEventListener('click', () => {
  const mode = document.querySelector('input[name="mode"]:checked').value
  const modeToEndpoint = {
    all: '/predict?include_cv=true&include_cv_audio=true&video_sample_every=8&cv_audio_sample_rate=0.25',
    nlp: '/test/nlp',
    cv: '/test/cv?sample_every=8',
    cv_audio: '/predict/cv-audio?sample_rate=0.25',
    deception: '/test/deception',
    emotion_av: '/test/emotion-av?sample_every=8&sample_rate=0.25',
    anomaly_audio: '/test/anomaly/audio',
    anomaly_video: '/test/anomaly/video?sample_every=8',
    anomaly_text: '/test/anomaly/text'
  }

  const endpoint = modeToEndpoint[mode]
  if(!endpoint){
    setStatus('Неизвестный режим: ' + mode)
    return
  }

  sendFileTo(endpoint)
})

clearBtn.addEventListener('click', clearResults)

clearResults()
