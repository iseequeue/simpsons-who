(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    screen: $('screen'), empty: $('empty'), view: $('view'), file: $('file'),
    pick: $('pick'), whole: $('whole'), hint: $('hint'), status: $('status'),
    answer: $('answer'), who: $('who'), whoMeta: $('whoMeta'), unsure: $('unsure'),
    topBlock: $('topBlock'), bars: $('bars'), alsoBlock: $('alsoBlock'), also: $('also'),
    cast: $('cast'), castSummary: $('castSummary'), footer: $('footer'),
    knob1: $('knob1'), knob2: $('knob2'),
  };
  const ctx = el.view.getContext('2d');

  const state = {
    blob: null,       // the uploaded file, re-sent with every request
    img: null,        // decoded image for drawing
    userBox: null,    // [x1,y1,x2,y2] in image pixels, drawn by the user
    dragging: null,   // {x0,y0,x1,y1} while drawing
    result: null,
    hoverAlso: -1,
    reqId: 0,
  };

  // ------------------------------------------------------------ helpers
  const pct = (p) => `${(p * 100).toFixed(p >= 0.995 ? 0 : 1).replace('.', ',')}%`;
  function setStatus(text, isError = false) {
    el.status.textContent = text;
    el.status.classList.toggle('error', isError);
  }

  // ------------------------------------------------------------ info
  fetch('/api/info').then((r) => r.json()).then((info) => {
    el.cast.replaceChildren(...info.classes.map((c) => Object.assign(document.createElement('li'), { textContent: c })));
    el.castSummary.textContent = `Кого знает модель (${info.classes.length})`;
    const acc = info.val?.acc;
    el.footer.textContent = `${info.model.split('.')[0]}, ${info.img}px` +
      (acc ? `, точность на отложенной выборке ${(acc * 100).toFixed(1).replace('.', ',')}%` : '') +
      `. Картинка обрабатывается на сервере и не сохраняется.`;
    setStatus('Готов. Жду скриншот.');
  }).catch(() => setStatus('Сервер не отвечает. Он запущен?', true));

  // ------------------------------------------------------------ drawing
  function draw() {
    const img = state.img;
    if (!img) return;
    const w = img.naturalWidth, h = img.naturalHeight;
    el.view.width = w;
    el.view.height = h;
    ctx.drawImage(img, 0, 0);
    const unit = Math.max(w, h) / 640;
    const r = state.result;

    if (r) {
      // where the main answer came from
      if (r.mode === 'box' || r.views > 1) box(r.box, '#ffd90f', `${r.top[0].name} ${pct(r.top[0].prob)}`, unit, false);
      // other characters, dashed, drawn last so their labels stay readable
      r.also.forEach((a, i) => box(a.box, i === state.hoverAlso ? '#ffd90f' : '#ffffff', `${a.name} ${pct(a.prob)}`, unit, true));
    }
    if (state.dragging) {
      const d = state.dragging;
      box([Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)], '#ffd90f', null, unit, true);
    }
  }

  function box(b, color, label, unit, dashed) {
    const [x1, y1, x2, y2] = b;
    const lw = Math.max(2, 3 * unit);
    ctx.save();
    ctx.lineWidth = lw;
    ctx.strokeStyle = color;
    ctx.setLineDash(dashed ? [8 * unit, 6 * unit] : []);
    ctx.shadowColor = 'rgba(0,0,0,.45)';
    ctx.shadowBlur = 4 * unit;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.restore();
    if (!label) return;
    const font = Math.round(Math.max(12, 15 * unit));
    ctx.font = `600 ${font}px Rubik, system-ui, sans-serif`;
    ctx.textBaseline = 'top';
    const pad = Math.round(font * 0.35);
    const tw = ctx.measureText(label).width + pad * 2;
    const th = font + pad * 2;
    const tx = Math.min(Math.max(0, x1), el.view.width - tw);
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = color;
    ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = '#111';
    ctx.fillText(label, tx + pad, ty + pad);
  }

  // canvas uses object-fit: contain, so map pointer → image pixels by hand
  function toImage(ev) {
    const rect = el.view.getBoundingClientRect();
    const iw = state.img.naturalWidth, ih = state.img.naturalHeight;
    const s = Math.min(rect.width / iw, rect.height / ih);
    const ox = rect.left + (rect.width - iw * s) / 2;
    const oy = rect.top + (rect.height - ih * s) / 2;
    return {
      x: Math.min(iw, Math.max(0, (ev.clientX - ox) / s)),
      y: Math.min(ih, Math.max(0, (ev.clientY - oy) / s)),
      s,
    };
  }

  // ------------------------------------------------------------ results
  function render(r) {
    state.result = r;
    state.hoverAlso = -1;
    const top = r.top[0];
    el.answer.hidden = false;
    el.who.textContent = top.name;
    const meta = [`уверенность ${pct(top.prob)}`];
    if (r.where && r.mode === 'auto') meta.push(r.where);
    if (r.mode === 'box') meta.push('в рамке');
    el.whoMeta.textContent = meta.join(', ');
    el.unsure.hidden = r.confident;

    el.topBlock.hidden = false;
    el.bars.replaceChildren(...r.top.map((t) => {
      const li = document.createElement('li');
      const name = Object.assign(document.createElement('span'), { textContent: t.name });
      const p = Object.assign(document.createElement('span'), { className: 'pct', textContent: pct(t.prob) });
      const track = Object.assign(document.createElement('div'), { className: 'track' });
      const fill = Object.assign(document.createElement('div'), { className: 'fill' });
      fill.style.width = `${Math.max(1, t.prob * 100)}%`;
      track.append(fill);
      li.append(name, p, track);
      return li;
    }));

    el.alsoBlock.hidden = r.also.length === 0;
    el.also.replaceChildren(...r.also.map((a, i) => {
      const li = document.createElement('li');
      li.tabIndex = 0;
      const left = document.createElement('span');
      left.append(a.name, ' ', Object.assign(document.createElement('small'), { textContent: a.where }));
      li.append(left, Object.assign(document.createElement('span'), { className: 'pct', textContent: pct(a.prob) }));
      const on = () => { state.hoverAlso = i; draw(); };
      const off = () => { state.hoverAlso = -1; draw(); };
      li.addEventListener('mouseenter', on); li.addEventListener('focus', on);
      li.addEventListener('mouseleave', off); li.addEventListener('blur', off);
      return li;
    }));

    el.knob1.style.setProperty('--turn', `${-135 + top.prob * 270}deg`);
    el.knob2.style.setProperty('--turn', `${-135 + Math.min(1, r.views / 4) * 270}deg`);
    const views = r.mode === 'box' ? 'рамка' : r.views > 1 ? `${r.views} части кадра` : 'весь кадр';
    setStatus(`Готово за ${r.ms} мс, ${views}.`);
    draw();
  }

  async function predict() {
    if (!state.blob) return;
    const id = ++state.reqId;
    const fd = new FormData();
    fd.append('image', state.blob, state.blob.name || 'screenshot.png');
    if (state.userBox) fd.append('box', state.userBox.map((v) => Math.round(v)).join(','));
    el.screen.classList.add('busy');
    setStatus(state.userBox ? 'Смотрю в рамку…' : 'Смотрю на кадр…');
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (id !== state.reqId) return; // a newer request superseded this one
      if (!res.ok) throw new Error(data.detail || `ошибка сервера ${res.status}`);
      render(data);
    } catch (e) {
      if (id === state.reqId) setStatus(`Не получилось: ${e.message}`, true);
    } finally {
      if (id === state.reqId) el.screen.classList.remove('busy');
    }
  }

  // ------------------------------------------------------------ input
  async function open(blob) {
    if (!blob || !blob.type.startsWith('image/')) {
      setStatus('Это не картинка: подойдут JPG, PNG или WebP.', true);
      return;
    }
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.src = url;
    try { await img.decode(); } catch {
      setStatus('Не получилось открыть картинку.', true);
      URL.revokeObjectURL(url);
      return;
    }
    if (state.img) URL.revokeObjectURL(state.img.src);
    Object.assign(state, { blob, img, userBox: null, result: null });
    el.empty.hidden = true;
    el.view.hidden = false;
    el.hint.hidden = false;
    el.whole.hidden = true;
    el.screen.classList.add('has-image');
    el.screen.setAttribute('aria-label', 'Скриншот. Обведи персонажа, чтобы спросить про него');
    el.pick.textContent = 'Другой скриншот';
    draw();
    predict();
  }

  el.pick.addEventListener('click', () => el.file.click());
  el.file.addEventListener('change', () => { if (el.file.files[0]) open(el.file.files[0]); el.file.value = ''; });
  el.whole.addEventListener('click', () => { state.userBox = null; el.whole.hidden = true; predict(); });

  el.screen.addEventListener('click', () => { if (!state.img) el.file.click(); });
  el.screen.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !state.img) { e.preventDefault(); el.file.click(); }
  });
  ['dragenter', 'dragover'].forEach((t) => el.screen.addEventListener(t, (e) => { e.preventDefault(); el.screen.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((t) => el.screen.addEventListener(t, () => el.screen.classList.remove('drag')));
  el.screen.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) open(f);
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith('image/'));
    if (item) open(item.getAsFile());
  });

  // draw a box around one character
  el.view.addEventListener('pointerdown', (e) => {
    if (!state.img) return;
    const p = toImage(e);
    state.dragging = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, s: p.s };
    el.view.setPointerCapture(e.pointerId);
  });
  el.view.addEventListener('pointermove', (e) => {
    if (!state.dragging) return;
    const p = toImage(e);
    state.dragging.x1 = p.x;
    state.dragging.y1 = p.y;
    draw();
  });
  el.view.addEventListener('pointerup', () => {
    const d = state.dragging;
    state.dragging = null;
    if (!d) return;
    const wPx = Math.abs(d.x1 - d.x0) * d.s, hPx = Math.abs(d.y1 - d.y0) * d.s; // on-screen size
    if (wPx < 12 || hPx < 12) { draw(); return; } // a click, not a box
    state.userBox = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
    const minSide = Math.min(state.userBox[2] - state.userBox[0], state.userBox[3] - state.userBox[1]);
    if (minSide < 16) { state.userBox = null; setStatus('Рамка слишком маленькая, обведи покрупнее.', true); draw(); return; }
    el.whole.hidden = false;
    predict();
  });
})();
