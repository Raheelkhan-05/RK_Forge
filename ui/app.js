// /ui/app.js

/* Forge UI. Plain JS, no build step, no network calls except to this server.
   All dynamic text is inserted with textContent / text nodes (never innerHTML) because model output is untrusted.

   Why typing can no longer be lost: the composer (textarea, buttons) lives in index.html and is NEVER re-created.
   Everything that refreshes (chat list, thread, status, task panel) re-renders its own container only, and the
   draft is also saved per chat in localStorage. */
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const h = (tag, props = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el[k] = v;
    else if (k === 'checked' || k === 'value' || k === 'disabled' || k === 'open') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
};

const STATUS = {queued:'Queued',preparing:'Preparing workspace',planning:'Planning',running:'Working',verifying:'Running checks',reviewing:'Reviewing',
  awaiting_user:'Needs your answer',awaiting_approval:'Ready for your approval',blocked:'Needs your decision',committing:'Applying changes',
  done:'Applied',failed:'Stopped, files unchanged',cancelled:'Cancelled',rejected:'Discarded',conflict:'Files changed meanwhile',interrupted:'Interrupted',paused:'Paused',undone:'Undone'};
const ACTIVE = new Set(['queued','preparing','planning','running','verifying','reviewing','committing','paused','awaiting_user','awaiting_approval','blocked']);
const WORKING = new Set(['running','planning','verifying','reviewing','preparing']);
const STAGES = ['Context','Code','Verify','Review','Accepted'];
const STAGE_IX = {context:0, code:1, verify:2, review:3, done:4};
const IDEAS = ['Walk me through how login works in this project.',
  'I’m thinking about renaming the status column to state. What would be affected?',
  'Add a “Remember me” checkbox to the login page and keep the user signed in through the API.'];

/* ------------------------------------------------------------------ state */
const S = {cfg: null, chats: [], hits: null, filter: '', cur: null, tab: 'task', facts: [], renaming: null, runInfo: {}, infoBusy: new Set(), loadingOlder: false, stick: true, attnKey: null, factFilter: ''};
const newChat = id => ({id, title: '', msgs: [], byId: new Map(), els: new Map(), hasMore: false, live: null, seq: 0, es: null, busy: false, queued: 0, loaded: false, retryT: 0});
const newRun = () => ({id: null, es: null, run: null, tasks: {}, order: [], feed: [], plan: null, pending: null, status: '', ctx: null, gates: [], diff: '', lastId: 0,
  live: {agent: '', task: '', nodes: []}, open: new Set(), replaying: false, retryT: 0});
let C = newChat(null);
const R = newRun();
const resetRun = () => Object.assign(R, newRun());

/* ------------------------------------------------------------------ api + helpers */
async function api(path, method = 'GET', body) {
  const res = await fetch(path, {method, headers: body !== undefined ? {'Content-Type': 'application/json'} : {}, body: body !== undefined ? JSON.stringify(body) : undefined});
  const text = await res.text();
  let data = text; try { data = JSON.parse(text); } catch (_) { /* plain text (diff) */ }
  if (!res.ok) { const d = data && (data.detail || data.error); throw new Error(typeof d === 'string' ? d : (d ? JSON.stringify(d) : res.statusText)); }
  return data;
}
function toast(msg) { const t = h('div', {class: 'toast', role: 'alert'}, msg); document.body.append(t); setTimeout(() => t.remove(), 4500); }
const fail = e => toast((e && e.message) || String(e));
function timeAgo(ts) {
  const s = Math.max(1, Math.floor(Date.now() / 1000 - ts));
  return s < 90 ? 'just now' : s < 5400 ? Math.round(s / 60) + ' min ago' : s < 129600 ? Math.round(s / 3600) + ' h ago' : Math.round(s / 86400) + ' d ago';
}
const clock = ts => new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
function groupLabel(ts) {
  const now = new Date(), day = 86400000, t = ts * 1000, start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  return t >= start ? 'Today' : t >= start - day ? 'Yesterday' : t >= start - 6 * day ? 'Previous 7 days' : t >= start - 29 * day ? 'Previous 30 days' : 'Older';
}
function copy(text, btn) {
  const done = () => { if (btn) { const old = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = old; }, 1200); } };
  const fallback = () => { const ta = h('textarea', {style: 'position:fixed;opacity:0'}); ta.value = text; document.body.append(ta); ta.select(); try { document.execCommand('copy'); done(); } catch (_) { /* ignore */ } ta.remove(); };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
}
const narrow = q => !!(window.matchMedia && window.matchMedia(q).matches);

/* ------------------------------------------------------------------ safe markdown -> DOM */
const RE = {
  fence: /^\s*```\s*([\w+#.-]*)\s*$/, fenceEnd: /^\s*```\s*$/, head: /^(#{1,4})\s+(.*)$/, hr: /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/,
  quote: /^\s*>\s?(.*)$/, li: /^(\s*)([-*+]|\d+[.)])\s+(.*)$/, tsep: /^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$/,
  inline: /(`+)([^`\n]+?)\1|\*\*([^*\n]+?)\*\*|\*([^*\s][^*\n]*?)\*|~~([^~\n]+?)~~|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])/g,
};
function pushText(parent, text) {
  text.split('\n').forEach((part, i) => { if (i) parent.append(document.createElement('br')); if (part) parent.append(document.createTextNode(part)); });
}
function link(url, label) { return h('a', {href: url, target: '_blank', rel: 'noopener noreferrer'}, label); }
function inline(text, parent) {
  const re = new RegExp(RE.inline.source, 'g');
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) pushText(parent, text.slice(last, m.index));
    if (m[2] != null) parent.append(h('code', {class: 'ic'}, m[2]));
    else if (m[3] != null) parent.append(h('strong', {}, m[3]));
    else if (m[4] != null) parent.append(h('em', {}, m[4]));
    else if (m[5] != null) parent.append(h('del', {}, m[5]));
    else if (m[6] != null) parent.append(link(m[7], m[6]));
    else parent.append(link(m[8], m[8]));
    last = re.lastIndex;
  }
  if (last < text.length) pushText(parent, text.slice(last));
}
function codeBlock(code, lang) {
  const btn = h('button', {class: 'btn ghost small', type: 'button', onclick: () => copy(code, btn)}, 'Copy');
  return h('div', {class: 'codeblock'}, h('div', {class: 'cb-head'}, h('span', {}, lang || 'code'), btn), h('pre', {}, h('code', {}, code)));
}
function isBlockStart(line, next) {
  return RE.fence.test(line) || RE.head.test(line) || RE.hr.test(line) || RE.quote.test(line) || RE.li.test(line) || (line.includes('|') && next !== undefined && RE.tsep.test(next));
}
function mdList(lines, i, out) {
  const stack = []; let lastLi = null;
  while (i < lines.length) {
    if (/^\s*$/.test(lines[i]) && i + 1 < lines.length && RE.li.test(lines[i + 1])) { i++; continue; }
    const m = RE.li.exec(lines[i]);
    if (!m) {
      if (lastLi && /^\s{2,}\S/.test(lines[i])) { lastLi.append(document.createElement('br')); inline(lines[i].trim(), lastLi); i++; continue; }
      break;
    }
    const indent = m[1].replace(/\t/g, '    ').length, ordered = /\d/.test(m[2]);
    while (stack.length && indent < stack[stack.length - 1].indent) stack.pop();
    let top = stack[stack.length - 1];
    if (!top || indent > top.indent) {
      const el = h(ordered ? 'ol' : 'ul');
      if (top && lastLi) lastLi.append(el); else out.append(el);
      top = {indent, el}; stack.push(top);
    }
    const li = h('li'); inline(m[3], li); top.el.append(li); lastLi = li; i++;
  }
  return i;
}
function md(src) {
  const out = document.createDocumentFragment();
  const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
  const cells = l => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]; let m;
    if (/^\s*$/.test(line)) { i++; continue; }
    if ((m = RE.fence.exec(line))) {
      const buf = []; i++;
      while (i < lines.length && !RE.fenceEnd.test(lines[i])) buf.push(lines[i++]);
      i++; out.append(codeBlock(buf.join('\n'), m[1])); continue;
    }
    if ((m = RE.head.exec(line))) { const hd = h('h' + (m[1].length + 2)); inline(m[2], hd); out.append(hd); i++; continue; }
    if (RE.hr.test(line)) { out.append(h('hr')); i++; continue; }
    if (RE.quote.test(line)) {
      const buf = [];
      while (i < lines.length && RE.quote.test(lines[i])) buf.push(RE.quote.exec(lines[i++])[1]);
      const bq = h('blockquote'); bq.append(md(buf.join('\n'))); out.append(bq); continue;
    }
    if (line.includes('|') && i + 1 < lines.length && RE.tsep.test(lines[i + 1])) {
      const head = cells(line), rows = []; i += 2;
      while (i < lines.length && lines[i].includes('|') && !/^\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      const cell = (tag, c) => { const el = h(tag); inline(c, el); return el; };
      out.append(h('table', {}, h('thead', {}, h('tr', {}, head.map(c => cell('th', c)))), h('tbody', {}, rows.map(r => h('tr', {}, r.map(c => cell('td', c)))))));
      continue;
    }
    if (RE.li.test(line)) { i = mdList(lines, i, out); continue; }
    const buf = [line]; i++;
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !isBlockStart(lines[i], lines[i + 1])) buf.push(lines[i++]);
    const p = h('p'); inline(buf.join('\n'), p); out.append(p);
  }
  return out;
}

/* ------------------------------------------------------------------ theme + header */
function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; }
try { applyTheme(localStorage.getItem('forge-theme')); } catch (_) { /* storage blocked */ }
$('#themeBtn').onclick = () => {
  const cur = document.documentElement.dataset.theme || (narrow('(prefers-color-scheme: dark)') ? 'dark' : 'light');
  const next = cur === 'dark' ? 'light' : 'dark'; try { localStorage.setItem('forge-theme', next); } catch (_) { /* ignore */ } applyTheme(next);
};
async function loadStatus() {
  try { S.cfg = await api('/api/status'); } catch (_) { return; }
  $('#repoPath').textContent = S.cfg.repo;
  const o = S.cfg.ollama, pill = $('#ollamaPill'), coder = S.cfg.roles.chat || S.cfg.roles.coder;
  pill.className = 'pill ' + (o.ok ? 'ok' : 'bad');
  const gpu = (o.placement || []).find(p => p.gpu_pct != null);
  pill.textContent = o.ok ? `${coder.model} · ${coder.num_ctx.toLocaleString()} ctx` + (gpu ? ` · ${gpu.gpu_pct}% GPU` : '') : (o.error || 'Ollama unavailable');
  pill.title = o.advice || ''; if (o.advice) pill.textContent += ' ⚠';
  const ix = S.cfg.index;
  $('#indexInfo').textContent = `${ix.files} files · ${ix.lines.toLocaleString()} lines · ${ix.projects.join(', ')}`;
  renderChatHeader();
  if (!$('#hero').hidden) paintHero();      // only the hero is repainted; the composer is never touched
}
function renderChatHeader() {
  const t = C.title || 'New chat';
  $('#chatTitle').textContent = t; document.title = (C.id ? t + ' · ' : '') + 'Forge';
  const auto = !S.cfg || S.cfg.chat.auto_start;
  $('#chatSub').textContent = auto ? 'Talk freely. A task starts when you say “go ahead”, or after you confirm its brief.' : 'Talk freely. A task only starts after you confirm its brief.';
}

/* ------------------------------------------------------------------ chat list */
async function loadChats() {
  try { S.chats = await api('/api/chats'); } catch (_) { return; }
  if (!S.filter) renderRail();
}
let loadChatsTimer = 0;
const loadChatsSoon = () => { clearTimeout(loadChatsTimer); loadChatsTimer = setTimeout(loadChats, 500); };
function renderRail() {
  if (S.renaming) return;
  const list = S.filter ? (S.hits || []) : S.chats, box = $('#chatList');
  box.replaceChildren();
  if (!list.length) { box.append(h('div', {class: 'empty small'}, S.filter ? 'No chats match.' : 'Your chats appear here.')); return; }
  let label = null;
  for (const c of list) {
    const g = S.filter ? null : groupLabel(c.updated);
    if (g && g !== label) { label = g; box.append(h('div', {class: 'group-label'}, g)); }
    box.append(chatItem(c));
  }
}
function chatItem(c) {
  const row = h('div', {class: 'ci' + (c.id === S.cur ? ' current' : ''), 'data-id': c.id});
  const draw = () => {
    row.replaceChildren(
      h('button', {class: 'open', onclick: () => selectChat(c.id), 'aria-current': c.id === S.cur ? 'true' : null, title: c.title},
        h('span', {class: 'dot ' + (c.attention ? 'attention' : c.running ? 'running' : '')}),
        h('span', {}, h('span', {class: 't'}, c.title || 'New chat'), h('span', {class: 'm'}, c.running ? (c.attention ? 'Needs you' : 'Task running') : (c.preview || timeAgo(c.updated))))),
      h('span', {class: 'acts'},
        h('button', {'aria-label': 'Rename chat', title: 'Rename', onclick: ev => { ev.stopPropagation(); startRename(c, row, draw); }}, '✎'),
        h('button', {'aria-label': 'Delete chat', title: 'Delete', onclick: ev => { ev.stopPropagation(); removeChat(c); }}, '×')));
  };
  draw();
  return row;
}
function startRename(c, row, redraw) {
  S.renaming = c.id;
  const input = h('input', {class: 'rename', value: c.title, 'aria-label': 'Chat name', maxlength: '80'});
  let done = false;
  const finish = async save => {
    if (done) return; done = true; S.renaming = null;
    const v = input.value.trim();
    if (save && v && v !== c.title) { try { await api(`/api/chats/${c.id}`, 'PUT', {title: v}); if (c.id === S.cur) { C.title = v; renderChatHeader(); } } catch (e) { fail(e); } }
    await loadChats(); renderRail();
  };
  input.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); finish(true); } else if (e.key === 'Escape') { e.preventDefault(); finish(false); } };
  input.onblur = () => finish(true);
  row.replaceChildren(input); input.focus(); input.select();
}
async function removeChat(c) {
  if (!confirm(`Delete “${c.title || 'New chat'}”? Its messages are removed. Facts saved to memory stay.`)) return;
  try { await api(`/api/chats/${c.id}`, 'DELETE', {}); } catch (e) { return fail(e); }
  if (c.id === S.cur) selectChat(null);
  loadChats();
}
let searchTimer = 0;
$('#chatSearch').addEventListener('input', e => {
  S.filter = e.target.value.trim(); clearTimeout(searchTimer);
  if (!S.filter) { renderRail(); return; }
  searchTimer = setTimeout(async () => { const q = S.filter; try { const hits = await api('/api/chats?q=' + encodeURIComponent(q)); if (q === S.filter) { S.hits = hits; renderRail(); } } catch (_) { /* ignore */ } }, 250);
});
$('#newChatBtn').onclick = () => { selectChat(null); $('#input').focus(); };
$('#reindexBtn').onclick = async () => { try { toast('Refreshing index…'); await api('/api/reindex', 'POST', {}); await loadStatus(); toast('Index is up to date'); } catch (e) { fail(e); } };

/* ------------------------------------------------------------------ drawers (narrow screens) */
function closeDrawers() { $('#rail').classList.remove('open'); $('#inspector').classList.remove('open'); $('#scrim').hidden = true; }
function toggleDrawer(sel) { const el = $(sel), open = !el.classList.contains('open'); closeDrawers(); if (open) { el.classList.add('open'); $('#scrim').hidden = false; } }
$('#railToggle').onclick = () => toggleDrawer('#rail');
$('#inspToggle').onclick = () => toggleDrawer('#inspector');
$('#scrim').onclick = closeDrawers;
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawers(); });
function openInspector(tab) { S.tab = tab || S.tab; renderInspector(); if (narrow('(max-width:1180px)')) { closeDrawers(); $('#inspector').classList.add('open'); $('#scrim').hidden = false; } }

/* ------------------------------------------------------------------ drafts + composer */
const input = $('#input');
const draftKey = () => 'forge-draft:' + (S.cur || 'new');
function saveDraft() { try { if (input.value.trim()) localStorage.setItem(draftKey(), input.value); else localStorage.removeItem(draftKey()); } catch (_) { /* ignore */ } }
function restoreDraft() { let v = ''; try { v = localStorage.getItem(draftKey()) || ''; } catch (_) { /* ignore */ } input.value = v; autosize(); }
function autosize() { input.style.height = 'auto'; input.style.height = Math.max(26, Math.min(220, input.scrollHeight)) + 'px'; }
let draftTimer = 0;
input.addEventListener('input', () => { autosize(); clearTimeout(draftTimer); draftTimer = setTimeout(saveDraft, 250); });
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send('chat'); } });
$('#sendBtn').onclick = () => send('chat');
$('#taskBtn').onclick = () => send('task');
$('#stopBtn').onclick = () => { if (C.id) api(`/api/chats/${C.id}/stop`, 'POST', {}).catch(fail); };

async function send(mode = 'chat', forced) {
  let text = (forced != null ? forced : input.value).trim();
  if (!text) return;
  if (/^\/task\s+\S/i.test(text)) { mode = 'task'; text = text.replace(/^\/task\s+/i, ''); }
  if (forced == null) { input.value = ''; autosize(); clearTimeout(draftTimer); try { localStorage.removeItem(draftKey()); } catch (_) { /* ignore */ } }
  try {
    if (!S.cur) { const conv = await api('/api/chats', 'POST', {}); selectChat(conv.id, {keepInput: true}); loadChats(); }
    const id = S.cur;
    const res = await api(`/api/chats/${id}/messages`, 'POST', {text, mode});
    if (C.id === id) { upsertMessage(res.message); S.stick = true; toBottom(true); }
  } catch (e) {
    if (forced == null) { input.value = input.value ? text + '\n' + input.value : text; autosize(); saveDraft(); }
    fail(e);
  }
}
const makeTask = () => send('brief', 'Turn what we discussed into a task.');

/* ------------------------------------------------------------------ selecting + connecting a chat */
function closeChatStream() { if (C.es) { C.es.close(); C.es = null; } clearTimeout(C.retryT); }
function selectChat(id, opts = {}) {
  if (id && id === S.cur && C.loaded) { closeDrawers(); return; }
  saveDraft(); closeChatStream(); closeRunStream(); resetRun(); resetTrace();
  S.cur = id; C = newChat(id); S.attnKey = null; S.stick = true;
  try { history.replaceState(null, '', id ? '#' + id : location.pathname + location.search); } catch (_) { /* ignore */ }
  if (!opts.keepInput) restoreDraft();
  $('#msgs').replaceChildren(); $('#liveHost').replaceChildren(); $('#older').replaceChildren();
  renderChatHeader(); updateEmpty(); renderRail(); renderInspector(); renderAttention(); updateStatus(); closeDrawers();
  if (id) connectChat(id);
}
function connectChat(id) {
  const es = new EventSource(`/api/chats/${id}/events`); C.es = es;
  es.onmessage = m => { let ev; try { ev = JSON.parse(m.data); } catch (_) { return; } if (C.id === id) onChatEvent(ev); };
  es.onerror = () => {
    if (C.es !== es || C.id !== id) return;
    es.close(); C.es = null; clearTimeout(C.retryT);
    C.retryT = setTimeout(() => { if (C.id === id && !C.es) connectChat(id); }, 1500);
  };
}
function applySnapshot(p) {
  C.title = p.conv.title; C.hasMore = !!p.has_more; C.seq = p.seq; C.loaded = true; C.queued = p.queued || 0;
  const ids = new Set(p.messages.map(m => m.id));
  if (p.messages.length) for (const m of [...C.msgs]) if (m.id >= p.messages[0].id && m.id <= p.max_id && !ids.has(m.id)) removeMessage(m.id);
  for (const m of p.messages) upsertMessage(m);
  C.live = p.live ? {text: p.live.text || '', phase: p.live.phase || '', reply_to: p.live.reply_to, activity: p.live.activity || []} : null;
  if (C.live && C.msgs.some(m => m.role === 'assistant' && m.kind === 'text' && m.meta && m.meta.reply_to === C.live.reply_to)) C.live = null;
  C.busy = !!C.live || !!p.busy;
  for (const e of p.trace || []) traceApply(e);
  renderChatHeader(); paintOlder(); paintLive(); updateEmpty(); updateLast(); updateStatus(); paintRunCards(); renderRail();
  S.stick = true; toBottom(true); autoAttach();
  if (S.tab === 'flow') paintFlow();
}
function onChatEvent(ev) {
  const p = ev.payload || {};
  if (ev.type === 'snapshot') return applySnapshot(p);
  if (ev.seq && ev.seq <= C.seq) return;
  if (ev.seq) C.seq = ev.seq;
  switch (ev.type) {
    case 'message': {
      upsertMessage(p.message);
      if (p.message.kind === 'run' && p.message.meta.run_id) attachRun(p.message.meta.run_id);
      if (p.message.kind === 'run_result') { delete S.runInfo[p.message.meta.run_id]; paintRunCards(); loadFacts(); }
      loadChatsSoon(); break;
    }
    case 'message_update': upsertMessage(p.message); break;
    case 'message_delete': removeMessage(p.id); break;
    case 'turn_start': C.busy = true; C.live = {text: '', phase: p.phase || 'Working', reply_to: p.reply_to, activity: []}; paintLive(); updateEmpty(); break;
    case 'phase': if (C.live) { C.live.phase = p.label; scheduleLive(); } break;
    case 'delta': if (C.live) { C.live.text += p.text; scheduleLive(); } break;
    case 'trace': traceApply(p); if (S.tab === 'flow') scheduleFlow(); break;
    case 'activity': if (C.live) { C.live.activity.push(p.label); C.live.phase = p.label; scheduleLive(); } break;
    case 'turn_end': C.busy = false; C.live = null; paintLive(); updateEmpty(); updateLast(); break;
    case 'queued': C.queued = p.count || 0; updateStatus(); break;
    case 'title': { C.title = p.title; const c = S.chats.find(x => x.id === C.id); if (c) c.title = p.title; renderChatHeader(); renderRail(); break; }
    case 'memory': if (S.tab === 'memory') loadFacts(); break;
    case 'run_status': delete S.runInfo[p.run_id]; paintRunCards(); loadChatsSoon(); break;
    case 'deleted': toast('This chat was deleted'); selectChat(null); loadChats(); break;
  }
}

/* ------------------------------------------------------------------ thread rendering */
const avatar = () => h('div', {class: 'avatar', 'aria-hidden': 'true'}, 'F');
const msgSig = m => `${m.id}|${(m.content || '').length}|${JSON.stringify(m.meta || {})}`;
function upsertMessage(m) {
  C.byId.set(m.id, m);
  const i = C.msgs.findIndex(x => x.id === m.id);
  if (i >= 0) C.msgs[i] = m; else { C.msgs.push(m); C.msgs.sort((a, b) => a.id - b.id); }
  const sig = msgSig(m), old = C.els.get(m.id);
  if (!old || old.dataset.sig !== sig) {
    const fresh = msgEl(m); fresh.dataset.sig = sig; fresh.dataset.mid = String(m.id);
    if (old) old.replaceWith(fresh);
    else {
      const next = C.msgs.find(x => x.id > m.id && C.els.get(x.id));
      if (next) $('#msgs').insertBefore(fresh, C.els.get(next.id)); else $('#msgs').append(fresh);
    }
    C.els.set(m.id, fresh);
  }
  if (m.role === 'assistant' && m.kind === 'text' && C.live && m.meta && m.meta.reply_to === C.live.reply_to) { C.live = null; paintLive(); }
  updateEmpty(); updateLast(); toBottom();
}
function removeMessage(id) {
  const el = C.els.get(id); if (el) el.remove();
  C.els.delete(id); C.byId.delete(id); C.msgs = C.msgs.filter(x => x.id !== id); updateEmpty(); updateLast();
}
function updateLast() {
  for (const el of $$('#msgs .msg.last')) el.classList.remove('last');
  if (C.live) return;
  for (let i = C.msgs.length - 1; i >= 0; i--) {
    const m = C.msgs[i];
    if (m.role === 'assistant' && m.kind === 'text') { const el = C.els.get(m.id); if (el) el.classList.add('last'); break; }
    if (m.role === 'user') break;
  }
}
function updateEmpty() {
  const hero = $('#hero'), empty = C.msgs.length === 0 && !C.live && (!S.cur || C.loaded), was = hero.hidden;
  hero.hidden = !empty;
  if (empty && (was || !hero.firstChild)) paintHero();
}
function paintHero() {
  const hero = $('#hero'), o = S.cfg && S.cfg.ollama;
  hero.replaceChildren(
    h('h2', {}, 'What are we working on?'),
    h('p', {}, 'Think it through first: ask how something works, compare options, or describe a change. I look at your real code, remember what you decide, and only touch your files after you confirm a task.'),
    h('div', {class: 'ideas'}, IDEAS.map(x => h('button', {onclick: () => { input.value = x; autosize(); input.focus(); saveDraft(); }}, x))),
    o && o.ok && o.advice ? h('div', {class: 'setup', role: 'status'}, o.advice) : null,
    o && !o.ok ? h('div', {class: 'setup', role: 'alert'}, h('b', {}, 'Ollama needs attention. '), o.error || '', h('div', {class: 'small'}, 'Start Ollama, then run ', h('code', {}, 'ollama pull ' + (S.cfg.roles.chat ? S.cfg.roles.chat.model : 'qwen3.5:9b')), '.')) : null);
}
function paintOlder() {
  const box = $('#older'); box.replaceChildren();
  if (C.hasMore) box.append(h('button', {class: 'btn small', style: 'margin:6px 0 14px', onclick: loadOlder}, 'Load earlier messages'));
}
async function loadOlder() {
  if (S.loadingOlder || !C.msgs.length) return;
  S.loadingOlder = true; const id = C.id;
  try {
    const r = await api(`/api/chats/${id}/messages?before=${C.msgs[0].id}&limit=100`);
    if (C.id !== id) return;
    const sc = $('#scroller'), before = sc.scrollHeight; S.stick = false;
    for (const m of r.messages) upsertMessage(m);
    C.hasMore = !!r.has_more; paintOlder(); sc.scrollTop += sc.scrollHeight - before;
  } catch (e) { fail(e); } finally { S.loadingOlder = false; }
}

function msgEl(m) {
  switch (m.kind) {
    case 'text': return m.role === 'user' ? userEl(m) : assistantEl(m, false);
    case 'brief': return briefEl(m);
    case 'run': return runEl(m);
    case 'run_result': return assistantEl(m, true);
    default: return noticeEl(m);
  }
}
function userEl(m) {
  const mode = (m.meta || {}).mode;
  return h('div', {class: 'msg user'}, h('div', {class: 'bubble'}, m.content), h('span', {class: 'time'}, (mode === 'task' ? 'Run as task, ' : mode === 'brief' ? 'Make a task, ' : '') + clock(m.ts)));
}
function chips(label, items) { return h('details', {class: 'chips'}, h('summary', {}, label), h('ul', {}, items.map(t => h('li', {}, t)))); }
function assistantEl(m, result) {
  const meta = m.meta || {};
  const body = h('div', {class: 'body'}, h('div', {class: 'md'}, md(m.content)));
  if (meta.stopped) body.append(h('div', {class: 'note-line'}, 'Stopped before it finished.'));
  if (meta.truncated) body.append(h('div', {class: 'note-line'}, 'The reply hit the length limit. Say “continue” to get the rest.'));
  if (meta.activity && meta.activity.length) body.append(chips(`Looked at the code (${meta.activity.length})`, meta.activity));
  if (meta.memory && meta.memory.length) body.append(chips(`Used ${meta.memory.length} saved ${meta.memory.length === 1 ? 'memory' : 'memories'}`, meta.memory.map(x => x.text)));
  const acts = h('div', {class: 'actions'}, h('button', {class: 'btn ghost small', onclick: ev => copy(m.content, ev.target)}, 'Copy'));
  if (!result) acts.append(h('button', {class: 'btn ghost small act-task', onclick: makeTask}, 'Make this a task'));
  body.append(acts);
  return h('div', {class: 'msg assistant' + (result ? ' result' : '')}, avatar(), body);
}
function noticeEl(m) {
  const meta = m.meta || {};
  const row = h('div', {class: 'row'}, h('div', {class: 'md'}, md(m.content)));
  if (meta.error && meta.retry_of) row.append(h('button', {class: 'btn small', onclick: () => api(`/api/chats/${C.id}/retry`, 'POST', {mid: meta.retry_of}).catch(fail)}, 'Retry'));
  return h('div', {class: 'msg notice' + (meta.error ? ' error' : '')}, row);
}
function briefEl(m) {
  const b = (m.meta || {}).brief || {}, st = (m.meta || {}).status || 'pending';
  const labels = {pending: 'Waiting for you', started: 'Started', dismissed: 'Not started', superseded: 'Replaced by a newer brief'};
  const req = h('div', {class: 'req'}, b.request || '');
  const card = h('div', {class: 'brief ' + st},
    h('div', {class: 'brief-head'}, h('span', {class: 'tag'}, 'Task brief'), h('b', {}, b.title || ''), st !== 'pending' ? h('span', {class: 'chip ' + st}, labels[st] || st) : null), req);
  const section = (title, items, cls) => { if (items && items.length) card.append(h('h4', {}, title), h('ul', {class: cls}, items.map(x => h('li', {}, x)))); };
  section('Must keep working', b.keep_working); section('Assumptions I made', b.assumptions); section('Still open', b.open_questions, 'open');
  if (st === 'pending') {
    const ta = h('textarea', {'aria-label': 'Edit the task', hidden: true}); ta.value = b.request || '';
    const start = h('button', {class: 'btn primary', onclick: ev => { ev.target.disabled = true; briefAct('start', ta.hidden ? null : ta.value).finally(() => { ev.target.disabled = false; }); }}, 'Start task');
    const edit = h('button', {class: 'btn', onclick: () => { ta.hidden = !ta.hidden; req.hidden = !ta.hidden; edit.textContent = ta.hidden ? 'Edit' : 'Cancel edit'; if (!ta.hidden) ta.focus(); }}, 'Edit');
    card.append(ta, h('div', {class: 'actions-row'}, start, edit, h('button', {class: 'btn ghost', onclick: () => briefAct('cancel')}, 'Not yet')));
  }
  return h('div', {class: 'msg assistant'}, avatar(), h('div', {class: 'body'}, card));
}
function briefAct(action, request) { return api(`/api/chats/${C.id}/brief`, 'POST', {action, request: request || undefined}).catch(fail); }

/* run card inside the thread */
function runEl(m) {
  const meta = m.meta || {};
  const card = h('div', {class: 'runcard', 'data-run': meta.run_id, 'data-title': meta.title || 'Task'});
  paintRunCard(card);
  return h('div', {class: 'msg assistant'}, avatar(), h('div', {class: 'body'}, card));
}
function ensureRunInfo(rid) {
  if (S.runInfo[rid] || S.infoBusy.has(rid)) return;
  S.infoBusy.add(rid);
  api('/api/runs/' + rid).then(r => { S.runInfo[rid] = r; }).catch(() => { S.runInfo[rid] = {status: 'unknown', tasks: []}; }).finally(() => { S.infoBusy.delete(rid); paintRunCards(); });
}
function runView(rid) {
  if (R.id === rid) {
    const ids = R.order, running = ids.map(i => R.tasks[i]).find(t => t.status === 'running');
    return {status: R.status || (R.run && R.run.status) || 'queued', done: ids.filter(i => R.tasks[i].status === 'accepted').length, total: ids.length, current: running ? running.title : ''};
  }
  const info = S.runInfo[rid];
  if (!info) { ensureRunInfo(rid); return {status: null, done: 0, total: 0, current: ''}; }
  const tasks = (info.tasks || []).filter(t => t.status !== 'replaced');
  return {status: info.status, done: tasks.filter(t => t.status === 'accepted').length, total: tasks.length, current: (tasks.find(t => t.status === 'running') || {}).title || ''};
}
function paintRunCard(card) {
  const rid = card.dataset.run, v = runView(rid), st = v.status, mine = R.id === rid;
  const kids = [h('div', {class: 'head'}, h('b', {}, card.dataset.title), st ? h('span', {class: 'chip ' + st}, STATUS[st] || st) : h('span', {class: 'dim small'}, 'Loading…'))];
  if (v.total) kids.push(h('div', {class: 'bar'}, h('i', {style: `width:${Math.round(v.done / v.total * 100)}%`})),
    h('div', {class: 'line'}, `${v.done} of ${v.total} steps accepted` + (v.current && ACTIVE.has(st) ? `, now: ${v.current}` : '')));
  else if (st === 'queued') kids.push(h('div', {class: 'line'}, 'Waiting for the current task to finish.'));
  const acts = h('div', {class: 'actions-row'}, h('button', {class: 'btn small', onclick: () => openRun(rid)}, 'Details'));
  if (mine && WORKING.has(st)) acts.append(h('button', {class: 'btn small', onclick: () => ctl('pause')}, 'Pause'));
  if (mine && st === 'paused') acts.append(h('button', {class: 'btn small primary', onclick: () => ctl('continue')}, 'Continue'));
  if (mine && ACTIVE.has(st) && !['awaiting_approval', 'committing'].includes(st)) acts.append(h('button', {class: 'btn small danger', onclick: () => confirm('Cancel this task? Your files are not touched either way.') && ctl('cancel')}, 'Cancel'));
  if (st === 'done') acts.append(h('button', {class: 'btn small', onclick: () => undoRun(rid)}, 'Undo'));
  if (st === 'interrupted') acts.append(h('button', {class: 'btn small primary', onclick: () => resumeRun(rid)}, 'Resume'));
  if (st === 'awaiting_user' || st === 'awaiting_approval' || st === 'blocked') kids.push(h('div', {class: 'line'}, 'This is waiting for you in the panel above the message box.'));
  kids.push(acts);
  card.replaceChildren(...kids);
}
function paintRunCards() { for (const c of $$('.runcard')) paintRunCard(c); }
async function undoRun(rid) {
  if (!confirm('Restore the previous version of the files this task changed?')) return;
  try { await api(`/api/runs/${rid}/undo`, 'POST', {}); toast('Restored the previous files'); delete S.runInfo[rid]; paintRunCards(); } catch (e) { fail(e); }
}
async function resumeRun(rid) {
  try { await api(`/api/runs/${rid}/control`, 'POST', {action: 'resume'}); delete S.runInfo[rid]; closeRunStream(); resetRun(); attachRun(rid); } catch (e) { fail(e); }
}
function openRun(rid) { attachRun(rid); openInspector('task'); }
function autoAttach() {
  const last = [...C.msgs].reverse().find(m => m.kind === 'run');
  if (!last) return;
  const rid = last.meta.run_id;
  api('/api/runs/' + rid).then(r => { if (C.id && C.msgs.includes(last) && ACTIVE.has(r.status) && !R.id) attachRun(rid); }).catch(() => {});
}

/* live streaming bubble + status line */
let liveRaf = 0;
function scheduleLive() { if (!liveRaf) liveRaf = requestAnimationFrame(() => { liveRaf = 0; paintLive(); }); }
function paintLive() {
  const host = $('#liveHost');
  if (!C.live) { if (host.firstChild) host.replaceChildren(); updateLast(); updateStatus(); return; }
  let el = host.firstChild;
  if (!el) {
    el = h('div', {class: 'msg assistant live'}, avatar(), h('div', {class: 'body'}, h('div', {class: 'md'}), h('div', {class: 'phase'}, h('span', {class: 'spinner'}), h('span', {class: 'ptext'}))));
    host.append(el);
  }
  if (el._text !== C.live.text) { el._text = C.live.text; el.querySelector('.md').replaceChildren(md(C.live.text)); }
  el.querySelector('.ptext').textContent = C.live.phase || 'Working';
  el.querySelector('.phase').hidden = !!C.live.text;
  updateLast(); updateStatus(); toBottom();
}
function updateStatus() {
  const bar = $('#statusLine'), waiting = Math.max(0, C.queued - (C.busy ? 1 : 0)), show = C.busy || C.queued > 0;
  bar.hidden = !show;
  if (!show) return;
  const phase = (C.live && C.live.phase) || (C.busy ? 'Working' : 'Waiting for the model');
  $('#statusText').textContent = phase + (waiting > 0 ? ` (${waiting} more waiting)` : '');
}

/* scrolling */
const scroller = $('#scroller'), jump = $('#jumpBtn');
scroller.addEventListener('scroll', () => { const near = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 90; S.stick = near; jump.hidden = near; });
function toBottom(force) { if (force || S.stick) requestAnimationFrame(() => { scroller.scrollTop = scroller.scrollHeight; }); }
jump.onclick = () => { S.stick = true; toBottom(true); jump.hidden = true; };

/* ------------------------------------------------------------------ run (task) state from events */
function closeRunStream() { if (R.es) { R.es.close(); R.es = null; } clearTimeout(R.retryT); }
function attachRun(rid) {
  if (!rid || R.id === rid) return;
  closeRunStream(); resetRun(); R.id = rid; R.replaying = true; S.attnKey = null;
  renderInspector(); renderAttention(); paintRunCards();
  connectRun(rid);
}
function connectRun(rid) {
  const es = new EventSource(`/api/runs/${rid}/events?after=${R.lastId}`); R.es = es;
  es.onmessage = m => {
    let ev; try { ev = JSON.parse(m.data); } catch (_) { return; }
    if (R.id !== rid) return;
    if (ev.id) R.lastId = Math.max(R.lastId, ev.id);
    reduceRun(ev); if (!R.replaying) scheduleRender('run');
  };
  es.addEventListener('ready', async () => {
    if (R.id !== rid) return;
    R.replaying = false;
    try {
      R.run = await api(`/api/runs/${rid}`);
      if (R.run.pending) { const k = R.run.pending.kind; R.pending = {...R.run.pending, kind: k === 'clarification' ? 'question' : k}; }
      else if (R.pending && ['question', 'doubt', 'approval', 'blocker'].includes(R.pending.kind) && !['awaiting_user', 'awaiting_approval', 'blocked'].includes(R.run.status)) R.pending = null;
      if (!R.status) R.status = R.run.status;
    } catch (e) { fail(e); }
    scheduleDiff(true); scheduleRender('run');
  });
  es.onerror = () => {
    if (R.es !== es || R.id !== rid) return;
    es.close(); R.es = null; clearTimeout(R.retryT);
    R.retryT = setTimeout(() => { if (R.id === rid && !R.es) connectRun(rid); }, 1500);
  };
}
function taskOf(id) {
  if (!R.tasks[id]) { R.tasks[id] = {id, title: id, status: 'pending', attempts: 0, stage: null, files: [], criteria: [], description: ''}; R.order.push(id); }
  return R.tasks[id];
}
function say(agent, text, o = {}) { R.feed.push({agent, text, sub: o.sub, cls: o.cls, task: o.task || null}); if (R.feed.length > 900) R.feed.splice(0, 200); }
function absorbPlan(tasks) {
  for (const t of tasks) {
    const x = taskOf(t.id); x.title = t.title; x.description = t.description; x.criteria = t.acceptance_criteria || [];
    x.files = [...(t.files_to_modify || []), ...(t.files_to_create || [])]; x.deps = t.depends_on || [];
  }
}
function reduceRun(ev) {
  const p = ev.payload || {}, T = p.task;
  switch (ev.type) {
    case 'status':
      R.status = p.status;
      if (!['awaiting_user', 'awaiting_approval', 'blocked'].includes(p.status) && R.pending) R.pending = null;
      if (p.status === 'cancelled') say('system', 'Task cancelled. Your files were not touched.');
      if (p.status === 'rejected') say('system', 'Discarded. Your files were not touched.');
      if (!R.replaying) { delete S.runInfo[R.id]; }
      break;
    case 'workspace': say('system', `Isolated workspace ready: ${p.files} files, ${(p.lines || 0).toLocaleString()} lines.`, {sub: `Model: ${p.model ? p.model.coder : ''}`}); break;
    case 'agent': say(p.agent, p.message); break;
    case 'span': traceApply(p); if (!R.replaying && S.tab === 'flow') scheduleFlow(); break;
    case 'tool': say(p.agent, `${p.action}${p.arg ? ': ' + p.arg : ''}`, {sub: p.thought}); break;
    case 'plan': R.plan = {understanding: p.understanding}; absorbPlan(p.tasks); say('planner', `Plan ready with ${p.tasks.length} subtask${p.tasks.length === 1 ? '' : 's'}.`); break;
    case 'plan_update': absorbPlan(p.tasks); say('planner', 'Plan updated: a failing step was split into smaller ones.'); break;
    case 'plan_rejected': say('system', 'The plan did not pass validation; asking the planner to fix it.', {sub: (p.errors || []).slice(0, 3).join(' | '), cls: 'bad'}); break;
    case 'task': { const x = taskOf(p.id); x.title = p.title || x.title; x.status = p.status; x.attempts = p.attempts; x.summary = p.summary;
      if (p.status === 'running') x.stage = x.stage && x.stage !== 'done' ? x.stage : 'context';
      if (p.status === 'accepted') x.stage = 'done'; if (p.status === 'pending') x.stage = null; break; }
    case 'attempt': { taskOf(T).stage = 'context'; say('system', `Attempt ${p.attempt} of ${p.of}`, {task: T}); break; }
    case 'context': { R.ctx = p; taskOf(T).stage = 'code';
      say('context', `Gave the coder ${p.items.filter(i => i.path).length} files/snippets (${p.tokens.toLocaleString()} of ${p.budget.toLocaleString()} tokens).`, {task: T, sub: p.omitted.length ? `${p.omitted.length} left out to fit` : ''}); break; }
    case 'coder_reply': { taskOf(T).stage = 'code';
      if (p.status === 'NEED_INFO') say('coder', 'Not sure yet, asking before changing anything.', {task: T, sub: [...p.asks, ...p.lookups].slice(0, 4).join(' | ')});
      else say('coder', p.summary || 'Proposed a change.', {task: T, sub: `${p.edits} edit${p.edits === 1 ? '' : 's'}`}); break; }
    case 'doubt': say('context', p.answered ? `Resolved from the code: ${p.question}` : `Could not resolve from the code: ${p.question}`, {task: T, sub: p.answered ? p.answer : 'Asking you.', cls: p.answered ? 'good' : ''}); break;
    case 'grounding': say('context', 'The coder relied on something I cannot find in the code.', {task: T, sub: p.unresolved.join(', '), cls: 'bad'}); break;
    case 'apply': taskOf(T).stage = 'verify';
      say('gates', p.ok ? `Applied edits to ${p.files.join(', ')} in the workspace.` : 'Edits could not be applied.', {task: T, sub: [...(p.errors || []), ...(p.notes || [])].slice(0, 3).join(' | '), cls: p.ok ? '' : 'bad'});
      scheduleDiff(); break;
    case 'gate_start': if (T && R.tasks[T]) R.tasks[T].stage = 'verify'; break;
    case 'gate_result': R.gates.push({...p, task: T});
      if (!p.skipped) say('gates', `${p.name} (${p.project}): ${p.ok ? 'passed' : p.preexisting ? 'failed before this change too' : 'failed'}`, {task: T, cls: p.ok ? 'good' : p.preexisting ? '' : 'bad'}); break;
    case 'review': { if (R.tasks[T]) R.tasks[T].stage = 'review';
      say('reviewer', p.verdict === 'accept' ? 'Accepted.' : 'Asked for changes.', {task: T === 'final' ? null : T, sub: p.feedback || '', cls: p.verdict === 'accept' ? 'good' : 'bad'}); break; }
    case 'retry': say('system', 'Trying again with specific feedback.', {task: T, sub: p.reason, cls: 'bad'}); break;
    case 'question': R.pending = {...p, kind: p.kind === 'doubt' ? 'doubt' : 'question'}; say('system', 'Waiting for your answer.'); break;
    case 'answered': R.pending = null; break;
    case 'approval': R.pending = p; scheduleDiff(); break;
    case 'blocker': R.pending = p; break;
    case 'error': say('system', p.message, {cls: 'bad', sub: p.detail}); break;
    case 'committed': say('system', `Applied ${p.files.length} file${p.files.length === 1 ? '' : 's'} to your repository.`, {cls: 'good'}); scheduleDiff(); loadFacts(); break;
    case 'stream_start': R.live = {agent: p.agent, task: p.task || '', nodes: []}; livePaintRun(true); break;
    case 'stream': R.live.nodes.push([p.kind, p.text]); if (R.live.nodes.length > 400) R.live.nodes.splice(0, 100); livePaintRun(false, p); break;
    case 'stream_end': break;
  }
}
let raf = 0; const dirty = new Set();
function scheduleRender(...parts) {
  parts.forEach(p => dirty.add(p));
  if (!raf) raf = requestAnimationFrame(() => {
    raf = 0; const d = new Set(dirty); dirty.clear();
    if (d.has('run')) { paintRunCards(); renderAttention(); }
    renderInspector();
  });
}
function ctl(action, extra = {}) { return R.id ? api(`/api/runs/${R.id}/control`, 'POST', {action, ...extra}).catch(fail) : Promise.resolve(); }

/* ------------------------------------------------------------------ attention panel (questions, approval, blockers) */
function renderAttention() {
  const box = $('#attn'), p = R.id ? R.pending : null;
  const key = p ? JSON.stringify([R.id, p.kind, p.id, p.task, (p.files || []).length]) : '';
  if (key === S.attnKey) return;
  S.attnKey = key; box.replaceChildren();
  if (!p) return;
  if (p.kind === 'question' || p.kind === 'doubt') {
    const areas = p.questions.map(q => h('textarea', {'aria-label': q, placeholder: 'Your answer'}));
    box.append(h('div', {class: 'attn', role: 'group'},
      h('h3', {}, p.kind === 'doubt' ? 'The coder is unsure and could not find the answer in your code' : 'A few answers needed before the task continues'),
      h('p', {}, p.kind === 'doubt' ? `Nothing has been changed. ${p.context || ''} Paste a decision, a snippet or a file path; it is remembered for later changes.` : (p.context || 'Short answers are fine. Leave one empty to let Forge choose the safest option.')),
      p.questions.map((q, i) => h('div', {class: 'q'}, h('label', {}, q), areas[i])),
      h('div', {class: 'actions'},
        h('button', {class: 'btn primary', onclick: async ev => { ev.target.disabled = true; try { await api(`/api/runs/${R.id}/answer`, 'POST', {answers: areas.map(a => a.value)}); } catch (e) { fail(e); ev.target.disabled = false; } }}, 'Send answers'),
        h('button', {class: 'btn', title: 'Forge picks the safest option and tells you what it assumed', onclick: async ev => { ev.target.disabled = true; try { await api(`/api/runs/${R.id}/answer`, 'POST', {answers: p.questions.map(() => '')}); } catch (e) { fail(e); ev.target.disabled = false; } }}, 'Let Forge decide'))));
  } else if (p.kind === 'approval') {
    const notes = h('textarea', {placeholder: 'What should be different? Forge will plan the follow-up and re-check everything.', hidden: true, 'aria-label': 'Requested changes'});
    const force = h('input', {type: 'checkbox'});
    box.append(h('div', {class: 'attn' + (p.partial ? ' bad' : ''), role: 'group'},
      h('h3', {}, `Ready to apply ${p.files.length} file${p.files.length === 1 ? '' : 's'}`),
      h('p', {}, 'All checks passed on an isolated copy. Your repository has not been touched yet. Applying is all-or-nothing.'),
      h('div', {class: 'filelist'}, p.files.map(f => h('div', {}, h('span', {}, f.path), h('span', {}, h('span', {class: 'add'}, '+' + f.added), ' ', h('span', {class: 'del'}, '−' + f.removed))))),
      p.external && p.external.length ? h('p', {class: 'bad'}, `${p.external.length} other file(s) changed on disk during the run (${p.external.slice(0, 3).join(', ')}). If they overlap with this change the apply will be refused safely.`) : null,
      p.partial ? h('label', {class: 'check'}, force, 'Some steps were skipped. Apply this partial result anyway.') : null,
      notes,
      h('div', {class: 'actions'},
        h('button', {class: 'btn primary', onclick: () => ctl('approve', {force: force.checked})}, 'Apply to my files'),
        h('button', {class: 'btn', onclick: () => openInspector('diff')}, 'Review changes'),
        h('button', {class: 'btn', onclick: ev => { if (notes.hidden) { notes.hidden = false; notes.focus(); ev.target.textContent = 'Send change request'; } else ctl('changes', {text: notes.value}); }}, 'Request changes'),
        h('button', {class: 'btn danger', onclick: () => confirm('Discard everything from this task?') && ctl('reject')}, 'Discard'))));
  } else if (p.kind === 'blocker') {
    const hint = h('textarea', {placeholder: 'Optional: tell Forge what to do differently (a function to use, a file to look at, a decision).', 'aria-label': 'Hint'});
    box.append(h('div', {class: 'attn bad', role: 'group'},
      h('h3', {}, `${p.task} could not be completed safely`), h('p', {}, `“${p.title}” failed every attempt. Nothing has been written to your files.`),
      h('pre', {class: 'out'}, p.failure), hint,
      h('div', {class: 'actions'}, h('button', {class: 'btn primary', onclick: () => ctl('retry', {text: hint.value})}, 'Retry with my hint'),
        h('button', {class: 'btn', onclick: () => ctl('skip')}, 'Skip this step'), h('button', {class: 'btn danger', onclick: () => ctl('abort')}, 'Stop, keep my files unchanged'))));
  }
}

/* ------------------------------------------------------------------ inspector panes */
function stageClass(t, i) {
  if (t.status === 'accepted') return 'ok';
  if (t.status === 'failed' || t.status === 'replaced') return i === (STAGE_IX[t.stage] ?? 1) ? 'fail' : (i < (STAGE_IX[t.stage] ?? 1) ? 'done' : '');
  if (t.status === 'skipped' || t.status === 'pending' || !t.stage) return '';
  const ix = STAGE_IX[t.stage]; return i < ix ? 'done' : i === ix ? 'now' : '';
}
const agentName = a => ({planner: 'Planner', context: 'Context', coder: 'Coder', reviewer: 'Reviewer', gates: 'Checks', system: 'System', memory: 'Memory'}[a] || a);
function feedRow(f) {
  return h('div', {class: 'ev'}, h('span', {class: 'agent ' + f.agent}, agentName(f.agent)),
    h('div', {class: 'what'}, h('div', {class: f.cls || ''}, f.text), f.sub ? h('div', {class: 'sub'}, String(f.sub).slice(0, 400)) : null));
}
function renderTask(t) {
  const running = t.status === 'running', open = R.open.has(t.id) || running;
  const ticks = h('span', {class: 'tick', title: 'Attempts used'}, ...Array.from({length: Math.max(3, t.attempts)}, (_, i) => h('i', {class: i < t.attempts ? (t.status === 'accepted' && i === t.attempts - 1 ? 'good' : 'on') : ''})));
  const track = h('div', {class: 'track', 'aria-hidden': 'true'}, STAGES.map((s, i) => h('div', {class: 'st ' + stageClass(t, i)}, s)));
  const meta = t.status === 'skipped' || t.status === 'replaced' ? t.summary : t.files.join(', ');
  const d = h('details', {class: 'task' + (running ? ' active' : ''), open},
    h('summary', {}, h('div', {}, h('div', {class: 'tt'}, `${t.id}: ${t.title}`), h('div', {class: 'tm'}, meta || ' ')), ticks, track),
    h('div', {class: 'body'},
      t.description ? h('div', {class: 'dim'}, t.description) : null,
      t.criteria.length ? h('ul', {class: 'crit'}, t.criteria.map(c => h('li', {}, c))) : null,
      h('div', {class: 'feed', style: 'margin-top:8px'}, R.feed.filter(f => f.task === t.id).slice(-60).map(feedRow))));
  d.ontoggle = () => { if (d.open) R.open.add(t.id); else R.open.delete(t.id); };
  return d;
}
function paintTask() {
  const box = $('#paneTask');
  if (!R.id) return box.replaceChildren(h('div', {class: 'empty'}, 'When a task runs, every step, check and decision shows up here. Start one from the chat.'));
  const st = R.status || (R.run && R.run.status) || 'queued';
  const kids = [h('div', {class: 'sec-title'}, 'Task ', h('span', {class: 'chip ' + st}, STATUS[st] || st), ' ', h('span', {class: 'dim small', id: 'elapsed'}))];
  if (R.run && R.run.request) kids.push(h('div', {class: 'understanding'}, R.run.request.length > 500 ? R.run.request.slice(0, 500) + '…' : R.run.request));
  if (R.plan && R.plan.understanding) kids.push(h('div', {class: 'dim small', style: 'margin-bottom:10px'}, R.plan.understanding));
  if (R.order.length) { kids.push(h('div', {class: 'sec-title'}, `Plan: ${R.order.length} subtask${R.order.length === 1 ? '' : 's'}`)); for (const id of R.order) kids.push(renderTask(R.tasks[id])); }
  const general = R.feed.filter(f => !f.task).slice(-120);
  if (general.length) kids.push(h('div', {class: 'sec-title'}, 'Activity'), h('div', {class: 'feed'}, general.map(feedRow)));
  box.replaceChildren(...kids); tickElapsed();
}
function renderInspector() {
  for (const b of $$('#tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === S.tab));
  for (const s of $$('#panes section')) s.hidden = s.dataset.pane !== S.tab;
  ({task: paintTask, diff: paintDiff, flow: paintFlow, live: () => livePaintRun(true), context: paintContext, gates: paintGates, memory: paintMemory}[S.tab] || paintTask)();
}
$('#tabs').onclick = e => { const b = e.target.closest('button'); if (!b) return; S.tab = b.dataset.tab; if (S.tab === 'memory') loadFacts(); renderInspector(); };
let dt = 0;
function scheduleDiff(now) {
  if (!R.id) return; const id = R.id; clearTimeout(dt);
  dt = setTimeout(async () => { try { const d = await api(`/api/runs/${id}/diff`); if (R.id === id) R.diff = d; } catch (_) { if (R.id === id) R.diff = ''; } if (S.tab === 'diff') paintDiff(); }, now ? 0 : 500);
}
function paintDiff() {
  const box = $('#paneDiff'); box.replaceChildren();
  if (!R.diff.trim()) return box.append(h('div', {class: 'empty'}, R.id ? 'No changes in the workspace yet.' : 'The changes of a task appear here.'));
  const files = []; let cur = null;
  for (const line of R.diff.split('\n')) {
    if (line.startsWith('--- ')) { cur = {name: '', lines: [], from: line.slice(4)}; files.push(cur); continue; }
    if (!cur) continue;
    if (line.startsWith('+++ ')) { cur.name = line.slice(4).replace(/^b\//, ''); if (cur.name === '/dev/null') cur.name = cur.from.replace(/^a\//, '') + ' (deleted)'; continue; }
    cur.lines.push(line);
  }
  for (const f of files) {
    const add = f.lines.filter(l => l.startsWith('+')).length, del = f.lines.filter(l => l.startsWith('-')).length;
    box.append(h('details', {class: 'diff-file', open: files.length <= 4}, h('summary', {}, h('span', {}, f.name), h('span', {}, h('span', {class: 'add'}, '+' + add), ' ', h('span', {class: 'del'}, '−' + del))),
      h('pre', {}, f.lines.slice(0, 600).map(l => h('span', {class: l.startsWith('+') ? 'a' : l.startsWith('-') ? 'd' : l.startsWith('@@') ? 'h' : ''}, l || ' ')))));
  }
}
function livePaintRun(full, ev) {
  const box = $('#paneLive'); if (!box || S.tab !== 'live') return;
  let txt = $('#liveText');
  if (full || !txt) {
    box.replaceChildren(h('div', {class: 'live-meta'}, h('span', {class: 'agent ' + (R.live.agent || 'system'), style: 'padding:2px 10px'}, agentName(R.live.agent || 'system')), h('span', {class: 'dim small'}, R.live.task || (R.live.agent ? '' : 'Nothing is generating right now.'))), txt = h('div', {id: 'liveText'}));
    for (const [k, t] of R.live.nodes) txt.append(k === 'thinking' ? h('span', {class: 'think'}, t) : document.createTextNode(t));
  } else if (ev) txt.append(ev.kind === 'thinking' ? h('span', {class: 'think'}, ev.text) : document.createTextNode(ev.text));
  box.scrollTop = box.scrollHeight;
}
function paintContext() {
  const box = $('#paneContext'); box.replaceChildren(); const c = R.ctx;
  if (!c) return box.append(h('div', {class: 'empty'}, 'The exact code given to the coder appears here for each attempt.'));
  box.append(h('div', {}, h('b', {}, `${c.task}: ${c.tokens.toLocaleString()} of ${c.budget.toLocaleString()} tokens`)), h('div', {class: 'bar'}, h('i', {style: `width:${Math.min(100, c.tokens / c.budget * 100)}%`})));
  for (const i of c.items) box.append(h('div', {class: 'item'}, h('span', {class: 'p'}, i.path || i.reason), h('span', {class: 'mode'}, i.mode + ' · ' + i.tokens), i.path ? h('span', {class: 'dim small'}, i.reason) : null));
  if (c.omitted.length) box.append(h('div', {class: 'sec-title'}, 'Left out to fit the window'), ...c.omitted.map(o => h('div', {class: 'small dim'}, o)));
}
function paintGates() {
  const box = $('#paneGates'); box.replaceChildren();
  if (!R.gates.length) return box.append(h('div', {class: 'empty'}, 'Syntax, import, lint, type, test and build results appear here. They decide acceptance, not the model.'));
  for (const g of R.gates.slice().reverse().slice(0, 60)) {
    const cls = g.skipped ? 'skip' : g.ok ? 'pass' : g.preexisting ? 'skip' : 'fail';
    box.append(h('div', {class: 'gate ' + cls}, h('b', {}, `${g.name} · ${g.project}`), ' ', h('span', {class: 'dim small'}, `${g.task || ''} ${g.skipped ? 'skipped: ' + g.reason : g.ok ? 'passed' : g.preexisting ? 'already failing before' : 'failed'}${g.seconds ? ' · ' + g.seconds + 's' : ''}`),
      !g.ok && !g.skipped && g.output ? h('pre', {class: 'out'}, g.output) : null));
  }
}

/* memory pane: the add-note form is built once so typing in it is never lost; only the list is repainted */
function initMemoryPane() {
  const box = $('#paneMemory');
  const ta = h('textarea', {placeholder: 'Add something Forge should always remember (a decision, a convention, how you like answers)…', 'aria-label': 'New memory note', maxlength: '400'});
  const kind = h('select', {'aria-label': 'Kind'}, ['fact', 'preference', 'decision', 'project', 'goal', 'convention'].map(k => h('option', {value: k}, k)));
  const add = h('button', {class: 'btn small primary', onclick: async () => {
    const text = ta.value.trim(); if (text.length < 8) return toast('Write a few more words.');
    try { await api('/api/memory', 'POST', {text, kind: kind.value}); ta.value = ''; loadFacts(); } catch (e) { fail(e); }
  }}, 'Save note');
  const find = h('input', {type: 'search', placeholder: 'Filter memory', 'aria-label': 'Filter memory', style: 'width:100%;padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);margin-bottom:8px'});
  find.oninput = () => { S.factFilter = find.value.trim().toLowerCase(); paintFacts(); };
  box.replaceChildren(h('div', {class: 'addnote'}, ta, h('div', {class: 'r'}, kind, add)), find, h('div', {id: 'factList'}));
}
async function loadFacts() { try { S.facts = await api('/api/memory'); } catch (_) { return; } if (S.tab === 'memory') paintFacts(); }
function paintMemory() { paintFacts(); }
function paintFacts() {
  const box = $('#factList'); if (!box) return; box.replaceChildren();
  const list = S.facts.filter(f => !S.factFilter || f.text.toLowerCase().includes(S.factFilter) || f.kind.includes(S.factFilter));
  if (!list.length) return box.append(h('div', {class: 'empty'}, S.facts.length ? 'Nothing matches.' : 'Decisions, preferences and facts from your chats and tasks are kept here and recalled when relevant. Facts about changed files expire automatically.'));
  for (const f of list) {
    const from = f.source.startsWith('chat') ? 'from a chat' : f.source === 'user' ? 'added by you' : 'from a task';
    box.append(h('div', {class: 'fact'}, h('div', {}, h('div', {class: 'k'}, `${f.kind}, ${from}${f.stale ? ', outdated' : ''}`), h('div', {}, f.text), f.paths.length ? h('div', {class: 'small dim mono'}, f.paths.slice(0, 2).join(', ')) : null),
      h('button', {class: 'btn ghost small', onclick: async () => { await api('/api/memory/' + f.id, 'DELETE', {}).catch(fail); loadFacts(); }}, 'Forget')));
  }
}
function tickElapsed() {
  const el = $('#elapsed'); if (!el || !R.run) return;
  const st = R.status || R.run.status, end = ACTIVE.has(st) ? Date.now() / 1000 : (R.run.finished || Date.now() / 1000);
  const s = Math.max(0, Math.floor(end - R.run.created)); el.textContent = `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

/* ------------------------------------------------------------------ settings */
const dlg = $('#settings');
$('#settingsBtn').onclick = () => {
  if (!S.cfg) return;
  const base = S.cfg.roles.chat || S.cfg.roles.coder;
  $('#setProfile').value = ['quality', 'fast'].includes(S.cfg.profile) ? S.cfg.profile : 'custom';
  $('#customWrap').hidden = $('#setProfile').value !== 'custom';
  $('#setModel').value = base.model; $('#setCtx').value = String(base.num_ctx);
  $('#setThink').checked = S.cfg.roles.coder.think; $('#setThinkChat').checked = !!(S.cfg.roles.chat && S.cfg.roles.chat.think);
  $('#setAutoStart').checked = !!S.cfg.chat.auto_start; $('#setApproval').checked = S.cfg.require_approval;
  dlg.showModal();
};
$('#setProfile').onchange = e => { $('#customWrap').hidden = e.target.value !== 'custom'; };
$('#saveSettings').onclick = async ev => {
  ev.preventDefault();
  const p = $('#setProfile').value;
  const body = {num_ctx: +$('#setCtx').value, think: $('#setThink').checked, think_chat: $('#setThinkChat').checked, auto_start: $('#setAutoStart').checked, require_approval: $('#setApproval').checked};
  if (p === 'custom') body.model = $('#setModel').value.trim() || 'qwen3.5:9b'; else body.profile = p;
  try { await api('/api/settings', 'PUT', body); dlg.close(); await loadStatus(); toast('Saved. New settings apply to the next message.'); } catch (e) { fail(e); }
};

/* ------------------------------------------------------------------ pipeline: real telemetry */
const TR = {spans: new Map(), traces: [], sel: null, selSpan: null, skew: 0};
function resetTrace() { TR.spans = new Map(); TR.traces = []; TR.sel = null; TR.selSpan = null; }
const nowS = () => Date.now() / 1000 - TR.skew;
const dur = s => Math.max(0, (s.t1 || nowS()) - s.t0);
const fmt = x => x == null ? '–' : x < 1 ? Math.round(x * 1000) + ' ms' : x < 60 ? x.toFixed(1) + ' s' : Math.floor(x / 60) + 'm ' + String(Math.round(x % 60)).padStart(2, '0') + 's';
const AGN = {signals: 'Wording signals', router: 'Intent router', controller: 'Controller', context: 'Context & code', chat: 'Chat model', planner: 'Planner',
  coder: 'Coder', reviewer: 'Reviewer', gates: 'Checks', summarizer: 'Background model', llm: 'Model'};
const agn = a => AGN[a] || a;

function traceApply(ev) {
  if (ev.phase !== 'end' && Math.abs(Date.now() / 1000 - ev.t) < 10) TR.skew = Date.now() / 1000 - ev.t;
  let s = TR.spans.get(ev.id);
  if (!s) {
    if (ev.phase === 'update') return;
    s = {id: ev.id, t0: ev.phase === 'end' ? ev.t - (ev.dur || 0) : ev.t, t1: null};
    TR.spans.set(ev.id, s);
    if (ev.kind === 'trace' && !TR.traces.includes(ev.id)) TR.traces.push(ev.id);
  }
  const t0 = s.t0; Object.assign(s, ev); s.t0 = t0;
  if (ev.phase === 'end') s.t1 = ev.t;
}
let flowRaf = 0;
function scheduleFlow() { if (!flowRaf) flowRaf = requestAnimationFrame(() => { flowRaf = 0; if (S.tab === 'flow') paintFlow(); }); }
const anyRunning = () => [...TR.spans.values()].some(s => !s.t1);

function flowNodes(spans) {
  const byId = new Map(spans.map(s => [s.id, s])), nodes = [], idx = new Map();
  for (const s of spans) {
    if (!s.agent) continue;
    const p = byId.get(s.parent);
    let n = idx.get(s.agent);
    if (!n) { n = {agent: s.agent, spans: [], time: 0, running: false, err: false, model: '', gen: 0, genS: 0, out: ''}; idx.set(s.agent, n); nodes.push(n); }
    n.spans.push(s);
    if (!p || p.agent !== s.agent) n.time += dur(s);          // count nested time once
    if (!s.t1) n.running = true;
    if (s.error) n.err = true;
    if (s.kind === 'llm') { n.model = s.model || n.model; if (s.gen_tokens && s.gen_s) { n.gen += s.gen_tokens; n.genS += s.gen_s; } }
    if (s.response) n.out = s.response;
  }
  return nodes;
}
function fnodeEl(n) {
  const llm = n.spans.filter(s => s.kind === 'llm'), tps = n.genS ? (n.gen / n.genS).toFixed(1) + ' tok/s' : '';
  return h('button', {class: `fnode a-${n.agent}${n.running ? ' running' : ''}${n.err ? ' err' : ''}`, type: 'button',
    onclick: () => { TR.selSpan = (llm[llm.length - 1] || n.spans[n.spans.length - 1]).id; paintFlow(); }},
    h('b', {}, agn(n.agent)), h('span', {class: 'ft'}, fmt(n.time) + (n.running ? ' …' : '')),
    h('span', {class: 'fm'}, [n.model, tps, n.spans.length > 1 ? n.spans.length + ' steps' : ''].filter(Boolean).join(' · ') || ' '));
}
function detailEl(s) {
  const kv = [], add = (k, v) => { if (v != null && v !== '') kv.push(h('div', {}, h('span', {class: 'dim'}, k), h('b', {}, String(v)))); };
  add('Took', fmt(dur(s)));
  if (s.kind === 'llm') {
    add('Model', s.model); add('Context window', s.num_ctx); add('Thinking', s.think ? 'on' : 'off'); add('Structured JSON', s.structured ? 'yes' : 'no'); add('Attempt', s.attempt);
    if (s.t1 && s.wall_s != null && !s.error) {
      add('Waiting for Ollama', fmt(s.queue_s)); add('Model load', fmt(s.load_s));
      add('Reading prompt', `${fmt(s.prefill_s)} · ${s.prompt_tokens} tok${s.prefill_tps ? ' @ ' + s.prefill_tps + '/s' : ''}`);
      add('Writing answer', `${fmt(s.gen_s)} · ${s.gen_tokens} tok${s.gen_tps ? ' @ ' + s.gen_tps + '/s' : ''}`);
      add('Prompt (my estimate)', s.est_prompt_tokens + ' tok');
    } else if (!s.t1) add('Right now', s.live === 'generating' ? 'writing the answer' : 'waiting for the first token (loading / reading the prompt)');
  }
  const kids = [h('div', {class: 'sec-title'}, `${agn(s.agent)}: ${s.name}`), h('div', {class: 'mgrid'}, kv)];
  if (s.hint) kids.push(h('div', {class: 'verdict'}, s.hint));
  if (s.error) kids.push(h('pre', {class: 'out bad'}, s.error));
  if (s.thought) kids.push(h('div', {class: 'small dim'}, 'Thought: ' + s.thought));
  for (const m of s.prompt || []) kids.push(h('details', {}, h('summary', {class: 'small'}, `Sent to model · ${m.role} · ${m.chars.toLocaleString()} chars`),
    h('pre', {class: 'out'}, m.head + (m.chars > m.head.length ? '\n…' : ''))));
  if (s.response) kids.push(h('div', {class: 'small dim'}, s.kind === 'llm' ? 'Model replied:' : 'Output:'), h('pre', {class: 'out'}, s.response));
  return h('div', {}, kids);
}
function paintFlow() {
  const box = $('#paneFlow');
  const ids = TR.traces.filter(i => TR.spans.has(i)).sort((a, b) => TR.spans.get(a).t0 - TR.spans.get(b).t0);
  if (!ids.length) return box.replaceChildren(h('div', {class: 'empty'}, 'Send a message and this view shows, live, every step it passes through and how long each really takes. Every number comes from measurements, nothing is simulated.'));
  const id = TR.sel && TR.spans.has(TR.sel) ? TR.sel : ids[ids.length - 1];
  const root = TR.spans.get(id), spans = [...TR.spans.values()].filter(s => s.trace === id && s.kind !== 'trace').sort((a, b) => a.t0 - b.t0);
  const total = dur(root), live = !root.t1, nodes = flowNodes(spans);
  const pick = h('select', {class: 'trace-pick', 'aria-label': 'Choose a request', onchange: e => { TR.sel = e.target.value; TR.selSpan = null; paintFlow(); }},
    ids.slice().reverse().map(i => { const r = TR.spans.get(i); return h('option', {value: i, selected: i === id}, `${clock(r.t0)} · ${r.name}${r.text ? ': ' + r.text.slice(0, 36) : ''} · ${fmt(dur(r))}`); }));
  const kids = [h('div', {class: 'flow-head'}, pick, h('span', {class: 'chip ' + (live ? 'running' : 'done')}, live ? 'live' : 'finished'), h('span', {class: 'dim small'}, 'total ' + fmt(total)))];
  const worst = nodes.slice().sort((a, b) => b.time - a.time)[0];
  const slow = spans.filter(s => s.kind === 'llm' && s.t1 && s.wall_s).sort((a, b) => b.wall_s - a.wall_s)[0];
  if (worst && total > 0.05) kids.push(h('div', {class: 'verdict'}, `${agn(worst.agent)} used ${fmt(worst.time)} (${Math.round(worst.time / total * 100)}% of this request).`,
    slow ? ` Slowest model call: ${agn(slow.agent)}, ${fmt(slow.wall_s)}, mostly ${slow.bottleneck}. ${slow.hint}` : ''));
  const graph = h('div', {class: 'flow-graph'});
  nodes.forEach((n, i) => {
    if (i) { const prev = nodes[i - 1]; graph.append(h('div', {class: 'fedge' + (n.running ? ' live' : ''), title: prev.out ? prev.out.slice(0, 300) : ''}, h('i'), h('span', {}, prev.out ? prev.out.length + ' ch' : ''))); }
    graph.append(fnodeEl(n));
  });
  kids.push(graph);
  const depth = s => { let d = 0, p = TR.spans.get(s.parent); while (p && p.kind !== 'trace' && d < 6) { d++; p = TR.spans.get(p.parent); } return d; };
  const W = Math.max(total, 0.001);
  kids.push(h('div', {class: 'sec-title'}, 'Timeline (real timings)'),
    h('div', {class: 'wf-legend'}, h('span', {}, h('i', {class: 'q'}), 'waiting for Ollama'), h('span', {}, h('i', {class: 'l'}), 'loading model'), h('span', {}, h('i', {class: 'p'}), 'reading prompt'), h('span', {}, h('i', {class: 'g'}), 'writing')));
  for (const s of spans.slice(0, 120)) {
    const left = (s.t0 - root.t0) / W * 100, width = Math.max(0.6, dur(s) / W * 100);
    const bar = h('div', {class: `wbar a-${s.agent}${s.t1 ? '' : ' running'}${s.error ? ' err' : ''}`, style: `left:${left}%;width:${Math.min(width, 100 - left)}%`});
    if (s.kind === 'llm' && s.t1 && s.wall_s) for (const [k, v] of [['q', s.queue_s], ['l', s.load_s], ['p', s.prefill_s], ['g', s.gen_s]]) if (v) bar.append(h('i', {class: 'seg ' + k, style: `width:${v / s.wall_s * 100}%`}));
    kids.push(h('div', {class: 'wrow' + (TR.selSpan === s.id ? ' sel' : ''), onclick: () => { TR.selSpan = s.id; paintFlow(); }},
      h('span', {class: 'wl', style: `padding-left:${depth(s) * 10}px`, title: s.name}, s.name), h('div', {class: 'wt'}, bar), h('span', {class: 'wd'}, fmt(dur(s)))));
  }
  const sel = TR.selSpan && TR.spans.get(TR.selSpan);
  if (sel) kids.push(detailEl(sel));
  box.replaceChildren(...kids);
}
setInterval(() => { if (S.tab === 'flow' && anyRunning()) paintFlow(); }, 250);
setInterval(() => {                       // the status line names who is working, with the real elapsed time
  if (!C.busy) return;
  const s = [...TR.spans.values()].filter(x => !x.t1 && x.kind === 'llm').pop();
  if (s) $('#statusText').textContent = `${agn(s.agent)} · ${s.live === 'generating' ? 'writing' : 'waiting for first token'} · ${fmt(dur(s))}`;
}, 500);

/* ------------------------------------------------------------------ boot */
setInterval(tickElapsed, 1000);
setInterval(() => { if (!S.filter) loadChats(); }, 6000);
setInterval(loadStatus, 20000);
window.addEventListener('hashchange', () => { const id = location.hash.slice(1) || null; if (id !== S.cur) selectChat(id); });
initMemoryPane();
renderInspector(); updateEmpty(); renderChatHeader();
loadStatus().then(loadChats).then(() => {
  const id = location.hash.slice(1);
  if (id && S.chats.some(c => c.id === id)) selectChat(id);
  else { const busy = S.chats.find(c => c.running || c.attention); if (busy) selectChat(busy.id); else { restoreDraft(); } }
});
loadFacts();
})();