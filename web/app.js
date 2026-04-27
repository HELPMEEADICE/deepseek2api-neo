/**
 * DeepSeek2API Neo · Material Design 3 Dashboard
 */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ======== 格式化工具 ========
const fmtNum = (n) => {
  if (n == null || isNaN(n)) return '--';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e4) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
};

const fmtTime = (ts) => {
  const d = new Date(ts * 1000);
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

// ======== 主题色 ========
function themeColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    primary: s.getPropertyValue('--md-sys-color-primary').trim(),
    secondary: s.getPropertyValue('--md-sys-color-secondary').trim(),
    error: s.getPropertyValue('--md-sys-color-error').trim(),
    surfaceVar: s.getPropertyValue('--md-sys-color-surface-variant').trim(),
    onSurface: s.getPropertyValue('--md-sys-color-on-surface').trim(),
  };
}

// Input/output 专用色
const COLOR_IN_FLASH  = '#1565C0';  // 蓝色系 - 输入
const COLOR_OUT_FLASH = '#42A5F5';  // 浅蓝系 - 输出
const COLOR_IN_PRO    = '#C62828';  // 红色系 - 输入
const COLOR_OUT_PRO   = '#EF5350';  // 浅红系 - 输出

// ======== 全局状态 ========
let period = 0;
let chartTokens = null;
let chartRequests = null;

// ======== API ========
async function fetchJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${path} returned ${resp.status}`);
  return resp.json();
}

// ======== 概览卡片 ========
async function updateOverview() {
  try {
    const [ov, models] = await Promise.all([
      fetchJSON(`/api/stats/overview?days=${period}`),
      fetchJSON(`/api/stats/models?days=${period}`),
    ]);

    $('#ovRequests').textContent = fmtNum(ov.total_requests);
    $('#ovTokens').textContent = fmtNum(ov.total_tokens);

    // 总体输入/输出
    const totalIn = models.reduce((s, m) => s + (m.prompt_tokens || 0), 0);
    const totalOut = models.reduce((s, m) => s + (m.completion_tokens || 0), 0);
    $('#ovInOut').innerHTML = totalIn + totalOut > 0
      ? `<span class="inout-item inout-item--in">入 ${fmtNum(totalIn)}</span><span class="inout-item inout-item--out">出 ${fmtNum(totalOut)}</span>`
      : '';

    const flash = models.find(m => m.model_display === 'DeepSeek-V4-Flash') || { request_count:0, prompt_tokens:0, completion_tokens:0, total_tokens:0 };
    const pro   = models.find(m => m.model_display === 'DeepSeek-V4-Pro')   || { request_count:0, prompt_tokens:0, completion_tokens:0, total_tokens:0 };

    $('#ovFlashReq').textContent = fmtNum(flash.request_count);
    $('#ovFlashInOut').innerHTML = (flash.prompt_tokens + flash.completion_tokens) > 0
      ? `<span class="inout-item inout-item--in">入 ${fmtNum(flash.prompt_tokens)}</span><span class="inout-item inout-item--out">出 ${fmtNum(flash.completion_tokens)}</span>`
      : '';

    $('#ovProReq').textContent = fmtNum(pro.request_count);
    $('#ovProInOut').innerHTML = (pro.prompt_tokens + pro.completion_tokens) > 0
      ? `<span class="inout-item inout-item--in">入 ${fmtNum(pro.prompt_tokens)}</span><span class="inout-item inout-item--out">出 ${fmtNum(pro.completion_tokens)}</span>`
      : '';
  } catch (e) {
    console.error('[updateOverview]', e);
  }
}

// ======== 图表 ========
async function updateCharts() {
  try {
    const dailyData = await fetchJSON('/api/stats/daily?days=30');

    // 始终生成最近30天（含今天），无数据则填0
    const allDates = [];
    for (let i = 29; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      allDates.push(d.toISOString().slice(0, 10));
    }

    const dateMap = {};
    for (const d of allDates) dateMap[d] = {};
    for (const row of dailyData) {
      if (dateMap[row.date]) dateMap[row.date][row.model_display] = row;
    }

    const sortedDates = allDates;
    const labels = sortedDates.map(d => d.slice(5));

    // 输入/输出拆分
    const flashIn  = sortedDates.map(d => (dateMap[d]['DeepSeek-V4-Flash'] || {}).prompt_tokens || 0);
    const flashOut = sortedDates.map(d => (dateMap[d]['DeepSeek-V4-Flash'] || {}).completion_tokens || 0);
    const proIn    = sortedDates.map(d => (dateMap[d]['DeepSeek-V4-Pro']   || {}).prompt_tokens || 0);
    const proOut   = sortedDates.map(d => (dateMap[d]['DeepSeek-V4-Pro']   || {}).completion_tokens || 0);
    const flashReqs = sortedDates.map(d => (dateMap[d]['DeepSeek-V4-Flash'] || {}).request_count || 0);
    const proReqs   = sortedDates.map(d => (dateMap[d]['DeepSeek-V4-Pro']   || {}).request_count || 0);

    const tc = themeColors();

    const chartOpts = (yLabel) => ({
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          grid: { color: tc.surfaceVar + '80' },
          ticks: { color: tc.onSurface, maxTicksLimit: 10, font: { size: 11 } },
        },
        y: {
          beginAtZero: true,
          stacked: true,
          title: { display: true, text: yLabel, color: tc.onSurface },
          grid: { color: tc.surfaceVar + '80' },
          ticks: { color: tc.onSurface, callback: v => fmtNum(v), font: { size: 11 } },
        },
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: tc.onSurface, usePointStyle: true, padding: 16, font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.raw)}`,
          },
        },
      },
    });

    // Token 图表 —— 堆叠柱状图：4 系列（输入/输出 × Flash/Pro）
    if (chartTokens) chartTokens.destroy();
    chartTokens = new Chart($('#chartTokens'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Flash 输入', data: flashIn,  backgroundColor: COLOR_IN_FLASH,  stack: 'flash', borderWidth: 0, borderRadius: 0 },
          { label: 'Flash 输出', data: flashOut, backgroundColor: COLOR_OUT_FLASH, stack: 'flash', borderWidth: 0, borderRadius: 4 },
          { label: 'Pro 输入',   data: proIn,    backgroundColor: COLOR_IN_PRO,    stack: 'pro',   borderWidth: 0, borderRadius: 0 },
          { label: 'Pro 输出',   data: proOut,   backgroundColor: COLOR_OUT_PRO,   stack: 'pro',   borderWidth: 0, borderRadius: 4 },
        ],
      },
      options: chartOpts('Token'),
    });

    // 请求数图表
    if (chartRequests) chartRequests.destroy();
    chartRequests = new Chart($('#chartRequests'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'V4-Flash',
            data: flashReqs,
            borderColor: COLOR_IN_FLASH,
            backgroundColor: COLOR_IN_FLASH + '33',
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            pointHoverRadius: 5,
          },
          {
            label: 'V4-Pro',
            data: proReqs,
            borderColor: COLOR_IN_PRO,
            backgroundColor: COLOR_IN_PRO + '33',
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            pointHoverRadius: 5,
          },
        ],
      },
      options: { ...chartOpts('请求数'), scales: { ...chartOpts('请求数').scales, y: { ...chartOpts('请求数').scales.y, stacked: false } } },
    });
  } catch (e) {
    console.error('[updateCharts]', e);
  }
}

// ======== 账号表格 — 按模型分开展示 ========
function renderAccountTable(tbodyId, modelDisplay, data) {
  const tbody = $(`#${tbodyId}`);
  const filtered = data.filter(r => r.model_display === modelDisplay);
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty">暂无数据</td></tr>`;
    return;
  }
  // 按 total_tokens 降序
  filtered.sort((a, b) => (b.total_tokens || 0) - (a.total_tokens || 0));
  const maxTotal = Math.max(...filtered.map(r => r.total_tokens || 0));

  tbody.innerHTML = filtered.map(r => {
    const bw = maxTotal > 0 ? Math.max(2, (r.total_tokens / maxTotal) * 100) : 0;
    return `<tr>
      <td>${r.account_id || '（直接 token）'}</td>
      <td class="r">${fmtNum(r.request_count)}</td>
      <td class="r">${fmtNum(r.prompt_tokens)}</td>
      <td class="r">${fmtNum(r.completion_tokens)}</td>
      <td class="r">
        <span class="token-bar" style="width:${bw}px;background:${modelDisplay === 'DeepSeek-V4-Flash' ? COLOR_IN_FLASH : COLOR_IN_PRO}"></span>
        ${fmtNum(r.total_tokens)}
      </td>
    </tr>`;
  }).join('');
}

async function updateAccounts() {
  try {
    const data = await fetchJSON(`/api/stats/accounts?days=${period}`);
    renderAccountTable('accountFlashTbody', 'DeepSeek-V4-Flash', data);
    renderAccountTable('accountProTbody', 'DeepSeek-V4-Pro', data);
  } catch (e) {
    console.error('[updateAccounts]', e);
  }
}

// ======== 最近请求 ========
async function updateRecent() {
  try {
    const data = await fetchJSON('/api/stats/recent?limit=15');
    const tbody = $('#recentTbody');
    if (!data.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">暂无数据</td></tr>';
      return;
    }
    tbody.innerHTML = data.map(r => `<tr>
      <td>${fmtTime(r.timestamp)}</td>
      <td>${r.model_display}</td>
      <td>${r.account_id || '--'}</td>
      <td class="r">${fmtNum(r.prompt_tokens)}</td>
      <td class="r">${fmtNum(r.completion_tokens)}</td>
      <td class="r">${fmtNum(r.total_tokens)}</td>
    </tr>`).join('');
  } catch (e) {
    console.error('[updateRecent]', e);
  }
}

// ======== 刷新 ========
async function refreshAll() {
  const btn = $('#refreshBtn');
  btn.classList.add('spinning');
  try {
    await Promise.all([
      updateOverview(),
      updateCharts(),
      updateAccounts(),
      updateRecent(),
      updateActive(),
    ]);
  } catch (e) {
    console.error('[refreshAll]', e);
  } finally {
    setTimeout(() => btn.classList.remove('spinning'), 400);
  }
}

// ======== 活动连接 ========
async function updateActive() {
  try {
    const data = await fetchJSON('/api/stats/active');
    const count = data.active_connections || 0;
    const chip = $('#activeChip');
    const counter = $('#activeCount');
    counter.textContent = count;
    if (count > 0) {
      chip.classList.add('has-connections');
    } else {
      chip.classList.remove('has-connections');
    }
  } catch (e) {
    // ignore
  }
}

// ======== 时间段切换 ========
$$('.segmented-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.segmented-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    period = parseInt(btn.dataset.days);
    refreshAll();
  });
});

// ======== 启动 ========
$('#refreshBtn').addEventListener('click', refreshAll);

let autoRefreshTimer = null;
let activePollTimer = null;

function startAutoRefresh() {
  stopAutoRefresh();
  autoRefreshTimer = setInterval(refreshAll, 60000);
  activePollTimer = setInterval(updateActive, 2000);
}
function stopAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  if (activePollTimer) clearInterval(activePollTimer);
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopAutoRefresh();
  else { refreshAll(); startAutoRefresh(); }
});

refreshAll().then(() => startAutoRefresh());
