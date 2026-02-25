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
  setStatus('Готов.')
  if(chart){ chart.destroy(); chart = null }
}

function prettyNumber(n){ return (n===null||n===undefined)?'-':(Math.round(n*1000)/1000).toString() }

function renderFormatted(formatted){
  // formatted expected shape created by backend
  const overall = formatted.overall || formatted
  const models = formatted.models || {}

  // Text summary
  const lines = []
  lines.push(`<b>Общий результат</b>`)
  lines.push(`Успешно: ${overall.success}`)
  lines.push(`Предсказание: ${overall.prediction}`)
  lines.push(`Уровень риска: ${overall.risk_level || (overall.risk_score>=0.7?'high':'medium')}`)
  if(overall.ensemble) lines.push(`Ensemble score: ${prettyNumber(overall.ensemble.ensemble_score)}`)

  lines.push(`<hr><b>Модели</b>`)
  if(models.nlp){
    lines.push(`<b>NLP модель</b>`)
    lines.push(`Успешно: ${models.nlp.success}`)
    lines.push(`Предсказание: ${models.nlp.prediction}`)
    lines.push(`Уровень риска: ${models.nlp.risk_level}`)
    lines.push(`Вероятность: ${prettyNumber(models.nlp.probability)}`)
  }
  if(models.cv_audio){
    lines.push(`<b>CV+Audio модель</b>`)
    lines.push(`Успешно: ${models.cv_audio.success}`)
    lines.push(`Предсказание: ${models.cv_audio.prediction}`)
    lines.push(`Уровень риска: ${models.cv_audio.risk_level}`)
    lines.push(`Вероятность: ${prettyNumber(models.cv_audio.probability)}`)
  }

  textResult.innerHTML = lines.map(l=>`<div>${l}</div>`).join('')

  // Chart: show probabilities per model
  const labels = []
  const vals = []
  if(models.nlp && models.nlp.probability!=null){ labels.push('NLP'); vals.push(models.nlp.probability) }
  if(models.cv_audio && models.cv_audio.probability!=null){ labels.push('CV+Audio'); vals.push(models.cv_audio.probability) }
  if(overall.ensemble && overall.ensemble.ensemble_score!=null){ labels.push('Ensemble'); vals.push(overall.ensemble.ensemble_score) }

  if(labels.length>0){
    if(chart) chart.destroy()
    chart = new Chart(probChartEl.getContext('2d'),{
      type:'bar',
      data:{labels, datasets:[{label:'Probability',data:vals,backgroundColor:['#2563eb','#10b981','#f59e0b']} ]},
      options:{scales:{y:{beginAtZero:true, max:1}}}
    })
  }
}

function normalizeResponse(json){
  // support new formatted_result + raw_result, or legacy flat result
  if(json.formatted_result) return json.formatted_result
  if(json.formatted) return json.formatted
  // build a tiny formatted object from legacy shape
  return {
    overall:{
      success: json.success,
      prediction: json.prediction,
      risk_level: json.risk_level,
      risk_score: json.risk_score,
      ensemble: json.ensemble_result || null
    },
    models: {
      nlp: { success: json.success, prediction: json.prediction, risk_level: json.risk_level, probability: json.risk_score },
      cv_audio: json.cv_audio_result?{ success: json.cv_audio_result.success, prediction: json.cv_audio_result.prediction, risk_level: json.cv_audio_result.risk_level, probability: json.cv_audio_result.probability }:null
    },
    segments: json.segments || [],
    features_csv: json.features_csv || null,
    audio_path: json.audio_path || null,
    error: json.error || null
  }
}

async function sendFileTo(endpoint){
  if(!videoInput.files || videoInput.files.length===0){ setStatus('Выберите файл'); return }
  const file = videoInput.files[0]
  const fd = new FormData()
  fd.append('file', file)

  setStatus('Отправка на ' + apiBase + endpoint + '...')
  console.log('[sendFileTo] POST', apiBase + endpoint, 'file:', file.name)
  try{
    const res = await fetch(apiBase + endpoint, { method:'POST', body:fd })
    console.log('[sendFileTo] Response status:', res.status)
    if(!res.ok){ 
      const txt = await res.text()
      console.error('[sendFileTo] Error response:', txt)
      setStatus('Ошибка от сервера: '+res.status); 
      textResult.textContent = txt; 
      return 
    }
    const data = await res.json()
    setStatus('Готово')
    const formatted = normalizeResponse(data)
    renderFormatted(formatted)
  }catch(err){ 
    console.error('[sendFileTo] Fetch error:', err)
    setStatus('Ошибка: '+err.message); 
    textResult.textContent = 'Fetch error: ' + err.message + '\n\nПроверьте консоль браузера (F12).'
  }
}

runBtn.addEventListener('click', ()=>{
  const mode = document.querySelector('input[name="mode"]:checked').value
  if(mode==='all'){
    // /predict runs full pipeline (we configured it to skip video-model but keep CV+Audio)
    sendFileTo('/predict')
  }else if(mode==='nlp'){
    sendFileTo('/test/nlp')
  }else if(mode==='cv_audio'){
    sendFileTo('/predict/cv-audio')
  }
})

clearBtn.addEventListener('click', clearResults)

// initial
clearResults()
