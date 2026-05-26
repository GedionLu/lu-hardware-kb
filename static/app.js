// AI服务助手 — 前端交互

const STORAGE_KEY = 'hw_chat_history';
const MAX_HISTORY = 50;

// ── 聊天历史 (localStorage) ──
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch { return []; }
}
function saveHistory(messages) {
  try {
    const trimmed = messages.slice(-MAX_HISTORY);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch {}
}

let messages = loadHistory();
let loading = false;

// ── DOM helpers ──
const chatEl = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('btn');

function scrollBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function timeStr() {
  const d = new Date();
  return d.getHours().toString().padStart(2,'0') + ':' +
         d.getMinutes().toString().padStart(2,'0');
}

// ── Render ──
function render() {
  chatEl.querySelectorAll('.msg').forEach(el => el.remove());
  const emptyState = chatEl.querySelector('.empty-state');
  if (messages.length === 0) {
    if (emptyState) emptyState.style.display = '';
    return;
  }
  if (emptyState) emptyState.style.display = 'none';

  messages.forEach((m, i) => {
    const div = document.createElement('div');
    div.className = 'msg ' + m.role;

    if (m.role === 'bot' && m.meta) {
      const header = document.createElement('div');
      header.className = 'bot-header';
      header.textContent = `型号: ${m.meta.models || '通用'} | ${m.meta.intents || ''}`;
      div.appendChild(header);
    }

    if (m.segments) {
      let hasImages = false;
      m.segments.forEach(seg => {
        if (seg.type === 'text') {
          const p = document.createElement('div');
          p.textContent = seg.content;
          div.appendChild(p);
        } else if (seg.type === 'image' && seg.url) {
          hasImages = true;
          const img = document.createElement('img');
          img.src = seg.url;
          img.alt = seg.label || '配置码';
          img.loading = 'lazy';
          img.onerror = () => { img.style.display = 'none'; };
          img.onclick = () => openLightbox(seg.url);
          div.appendChild(img);
          if (seg.label) {
            const label = document.createElement('div');
            label.className = 'img-label';
            label.textContent = seg.label;
            div.appendChild(label);
          }
        }
      });
      if (hasImages) {
        const hint = document.createElement('div');
        hint.style.cssText = 'font-size:11px;color:#aaa;margin-top:4px;';
        hint.textContent = '💡 点击图片可放大查看';
        div.appendChild(hint);
      }
    } else if (m.content) {
      div.textContent = m.content;
    }

    // 全文检索结果
    if (m.fulltext && m.fulltext.length > 0) {
      const ftDiv = document.createElement('div');
      ftDiv.style.cssText = 'margin-top:10px;padding:10px;background:#f8f9fa;border-left:3px solid #4a90d9;border-radius:4px;font-size:13px;';
      const ftTitle = document.createElement('div');
      ftTitle.style.cssText = 'font-weight:bold;color:#4a90d9;margin-bottom:6px;';
      ftTitle.textContent = '📖 手册相关章节:';
      ftDiv.appendChild(ftTitle);
      m.fulltext.forEach(ft => {
        const item = document.createElement('div');
        item.style.cssText = 'margin:4px 0;color:#555;';
        item.textContent = `[${ft.product}] ${ft.chapter} (p${ft.page})`;
        ftDiv.appendChild(item);
      });
      div.appendChild(ftDiv);
    }

    const time = document.createElement('div');
    time.className = 'time';
    time.textContent = m.time || '';
    div.appendChild(time);

    chatEl.appendChild(div);
  });

  scrollBottom();
}

// ── Actions ──
function addUser(text) {
  messages.push({ role: 'user', content: text, time: timeStr() });
  saveHistory(messages);
  render();
}

function addBot(data, meta) {
  messages.push({
    role: 'bot',
    segments: data.segments || [],
    fulltext: data.fulltext || [],
    meta: meta || {},
    time: timeStr(),
  });
  saveHistory(messages);
  render();
}

function addError(text) {
  messages.push({ role: 'bot', content: text, time: timeStr(), error: true });
  saveHistory(messages);
  render();
}

function addLoading() {
  const div = document.createElement('div');
  div.className = 'msg bot loading';
  div.id = 'loading-msg';
  const span = document.createElement('span');
  span.className = 'dots';
  span.textContent = '查询中';
  div.appendChild(span);
  chatEl.appendChild(div);
  scrollBottom();
}

function removeLoading() {
  const el = document.getElementById('loading-msg');
  if (el) el.remove();
}

async function send() {
  if (loading) return;
  const text = inp.value.trim();
  if (!text) return;

  inp.value = '';
  btn.disabled = true;
  loading = true;
  addUser(text);
  addLoading();

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text }),
    });

    removeLoading();

    if (resp.status === 429) {
      addError('⚠️ 请求太频繁，请稍后再试');
    } else if (!resp.ok) {
      addError('⚠️ 查询失败 (HTTP ' + resp.status + ')');
    } else {
      const data = await resp.json();
      addBot(data, { models: data.models, intents: data.intents });
    }
  } catch (e) {
    removeLoading();
    addError('⚠️ 网络错误: ' + e.message);
  }

  loading = false;
  btn.disabled = false;
  inp.focus();
}

// ── Quick ask ──
function quickAsk(query) {
  inp.value = query;
  send();
}

// ── Clear ──
function clearChat() {
  if (messages.length === 0) return;
  if (confirm('确认清空对话记录？')) {
    messages = [];
    saveHistory(messages);
    render();
  }
}

// ── Lightbox ──
function openLightbox(url) {
  let lb = document.getElementById('lightbox');
  if (!lb) {
    lb = document.createElement('div');
    lb.id = 'lightbox';
    lb.className = 'lightbox';
    lb.onclick = () => lb.classList.remove('active');
    document.body.appendChild(lb);
    const img = document.createElement('img');
    img.id = 'lightbox-img';
    lb.appendChild(img);
    lb.addEventListener('click', (e) => {
      if (e.target === lb) lb.classList.remove('active');
    });
  }
  document.getElementById('lightbox-img').src = url;
  lb.classList.add('active');
}

// ── Keyboard ──
inp.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// ── Init ──
render();
