/**
 * LayerTimingChart — Canvas Gantt Chart (matching Python visualize_workload.py)
 *
 * Layout: 5 fixed rows — Compute, TP Comm, DP Comm, EP Comm, DP_EP Comm
 * X axis = time. Events rendered as colored blocks at their row.
 */

import { useRef, useCallback, useEffect, useState } from 'react';
import type { LayerResult } from '../../types/wizard';

/* ── colour palette ── */
const C = {
  bg: '#06080d',
  surface: '#0d1117',
  surface2: '#161b22',
  border: '#21262d',
  text: '#e6edf3',
  text2: '#8b949e',
  text3: '#484f58',

  // Compute (gray tones)
  compute_fwd: '#4b5563',
  compute_bwd: '#374151',

  // Comm by group (bright, saturated)
  tp:   '#3b82f6',  // blue
  dp:   '#f59e0b',  // amber
  ep:   '#10b981',  // emerald
  dp_ep:'#8b5cf6',  // violet
  pp:   '#ef4444',  // red

  // Row backgrounds (alternating)
  row0: 'rgba(255,255,255,0.02)',
  row1: 'rgba(255,255,255,0.04)',
  row2: 'rgba(255,255,255,0.02)',
  row3: 'rgba(255,255,255,0.04)',
  row4: 'rgba(255,255,255,0.02)',
};

const FONT = '11px JetBrains Mono, monospace';
const HEADER_H = 48;
const ROW_H = 36;
const ROW_GAP = 4;
const ROW_LABEL_W = 100;
const ROWS = ['Compute', 'TP Comm', 'DP Comm', 'EP Comm', 'DP_EP Comm'] as const;
const ROW_BG = [C.row0, C.row1, C.row2, C.row3, C.row4];
const NUM_ROWS = ROWS.length;
const ROW_GROUP_MAP: Record<string, number> = {
  NONE: 0, TP: 1, DP: 2, EP: 3, DP_EP: 4, PP: 4,
};

/* ── format helpers ── */
function fmtTime(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 's';
  if (v >= 1e3) return (v / 1e3).toFixed(2) + 'ms';
  return v.toFixed(0) + 'us';
}

function fmt(v: number): string {
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  return v.toFixed(0);
}

/* ── get comm group for a phase (now from backend data, not inferred from name) ── */
function getPhaseGroup(layer: LayerResult, phase: 'fwd' | 'wg' | 'ig'): string {
  if (phase === 'fwd') return layer.fwd_group ?? 'TP';
  if (phase === 'wg') return layer.wg_group ?? 'DP';
  return layer.ig_group ?? 'TP';
}

/* ── Gantt event ── */
interface GanttEvent {
  start: number;
  duration: number;
  color: string;
  phase: string;
  row: number;   // 0=Compute, 1=TP, 2=DP, 3=EP, 4=DP_EP
  name: string;
  layerIdx: number;
  isComm: boolean;
}

function buildGanttEvents(layers: readonly LayerResult[]): GanttEvent[] {
  const events: GanttEvent[] = [];
  let t = 0.0;

  // ── Forward pass ──
  for (let i = 0; i < layers.length; i++) {
    const l = layers[i];
    const fwdC = l.fwd_compute ?? 0;
    const fwdE = l.fwd_exposed_comm ?? 0;

    // Compute (row 0)
    if (fwdC > 0) {
      events.push({ start: t, duration: fwdC, color: C.compute_fwd, phase: 'fwd', row: 0, name: l.layer_name, layerIdx: i, isComm: false });
      t += fwdC;
    }
    // Comm
    if (fwdE > 0) {
      const g = getPhaseGroup(l, 'fwd');
      const color = g === 'DP' ? C.dp : g === 'EP' ? C.ep : g === 'DP_EP' ? C.dp_ep : C.tp;
      events.push({ start: t, duration: fwdE, color, phase: 'fwd', row: ROW_GROUP_MAP[g] ?? 1, name: l.layer_name, layerIdx: i, isComm: true });
      t += fwdE;
    }
  }

  // ── Backward pass: ig first, then wg ──
  for (let i = layers.length - 1; i >= 0; i--) {
    const l = layers[i];
    const igC = l.ig_compute ?? 0;
    const igE = l.ig_exposed_comm ?? 0;
    const wgC = l.wg_compute ?? 0;
    const wgE = l.wg_exposed_comm ?? 0;

    // Input gradient compute (row 0)
    if (igC > 0) {
      events.push({ start: t, duration: igC, color: C.compute_bwd, phase: 'bwd_ig', row: 0, name: l.layer_name, layerIdx: i, isComm: false });
      t += igC;
    }
    // Input gradient comm
    if (igE > 0) {
      const g = getPhaseGroup(l, 'ig');
      const color = g === 'DP' ? C.dp : g === 'EP' ? C.ep : g === 'DP_EP' ? C.dp_ep : C.tp;
      events.push({ start: t, duration: igE, color, phase: 'bwd_ig', row: ROW_GROUP_MAP[g] ?? 1, name: l.layer_name, layerIdx: i, isComm: true });
      t += igE;
    }

    // Weight gradient compute (row 0)
    if (wgC > 0) {
      events.push({ start: t, duration: wgC, color: C.compute_bwd, phase: 'bwd_wg', row: 0, name: l.layer_name, layerIdx: i, isComm: false });
      t += wgC;
    }
    // Weight gradient comm
    if (wgE > 0) {
      const g = getPhaseGroup(l, 'wg');
      const color = g === 'DP' ? C.dp : g === 'EP' ? C.ep : g === 'DP_EP' ? C.dp_ep : C.tp;
      events.push({ start: t, duration: wgE, color, phase: 'bwd_wg', row: ROW_GROUP_MAP[g] ?? 1, name: l.layer_name, layerIdx: i, isComm: true });
      t += wgE;
    }
  }

  return events;
}

interface Props {
  readonly layers: readonly LayerResult[];
  readonly runName?: string;
}

export function LayerTimingChart({ layers, runName }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const panRef = useRef(0);
  const dragRef = useRef({ startX: 0, startPan: 0 });

  const events = buildGanttEvents(layers);
  const totalTime = Math.max(...events.map(e => e.start + e.duration), 1);

  // FWD end for separator
  const fwdEndTime = (() => {
    let t = 0;
    for (const l of layers) {
      t += (l.fwd_compute ?? 0) + (l.fwd_exposed_comm ?? 0);
    }
    return t;
  })();

  const totalH = HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP) + 20;

  /* ── layout ── */
  const getLayout = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return null;

    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth;
    if (W === 0) return null;  // DOM not yet painted

    canvas.width = W * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = totalH + 'px';

    const drawW = W - ROW_LABEL_W;
    const contentW = drawW * zoom;
    const maxPan = Math.max(0, contentW - drawW);
    const pan = zoom <= 1 ? 0 : Math.max(0, Math.min(maxPan, panRef.current));

    return { canvas, wrap, dpr, W, totalH, pan, maxPan, drawW, contentW };
  }, [zoom, totalTime, totalH]);

  /* ── draw ── */
  const draw = useCallback(() => {
    const layout = getLayout();
    if (!layout) return;
    const { canvas, W, totalH, pan, drawW, contentW } = layout;
    const dpr = canvas.width / W;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, totalH);

    const scale = contentW / totalTime;  // pixels per time unit

    // ── Background ──
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, totalH);

    // ── Header ──
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, W, HEADER_H);
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, HEADER_H); ctx.lineTo(W, HEADER_H); ctx.stroke();

    ctx.font = '600 12px JetBrains Mono, monospace';
    ctx.fillStyle = C.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(`Layer Timeline — ${runName ?? 'Simulation'}`, 12, HEADER_H / 2);

    const totalCompute = layers.reduce((s, l) => s + (l.fwd_compute ?? 0) + (l.ig_compute ?? 0) + (l.wg_compute ?? 0), 0);
    const totalComm = layers.reduce((s, l) => s + (l.fwd_exposed_comm ?? 0) + (l.ig_exposed_comm ?? 0) + (l.wg_exposed_comm ?? 0), 0);
    ctx.font = FONT;
    ctx.fillStyle = C.text2;
    ctx.textAlign = 'right';
    ctx.fillText(`Compute: ${fmt(totalCompute)}  Comm: ${fmt(totalComm)}  Total: ${fmtTime(totalTime)}`, W - 12, HEADER_H / 2);

    // ── Row labels (left column) ──
    ctx.font = '600 10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ROWS.forEach((label, i) => {
      const y = HEADER_H + i * (ROW_H + ROW_GAP);
      // Row background
      ctx.fillStyle = ROW_BG[i];
      ctx.fillRect(0, y, ROW_LABEL_W, ROW_H);
      // Label
      ctx.fillStyle = C.text2;
      ctx.fillText(label, ROW_LABEL_W - 8, y + ROW_H / 2);
    });

    // Scrollable content area clip
    ctx.save();
    ctx.beginPath();
    ctx.rect(ROW_LABEL_W, HEADER_H, drawW, totalH - HEADER_H);
    ctx.clip();

    // ── Row backgrounds (scoped) ──
    ROWS.forEach((_, i) => {
      const y = HEADER_H + i * (ROW_H + ROW_GAP);
      ctx.fillStyle = ROW_BG[i];
      ctx.fillRect(ROW_LABEL_W, y, drawW, ROW_H);
    });

    // ── FWD/BWD separator ──
    if (fwdEndTime > 0) {
      const sepX = ROW_LABEL_W + fwdEndTime * scale - pan;
      if (sepX >= ROW_LABEL_W && sepX <= W) {
        ctx.strokeStyle = 'rgba(248,81,73,0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sepX, HEADER_H);
        ctx.lineTo(sepX, HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP));
        ctx.stroke();
        ctx.font = '600 9px JetBrains Mono, monospace';
        ctx.fillStyle = '#f85149';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('\u2190 FWD | BWD \u2192', sepX, HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP) + 4);
      }
    }

    // ── Time axis ticks ──
    const ticks = 6;
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = C.text3;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= ticks; i++) {
      const tx = (i / ticks) * totalTime;
      const x = ROW_LABEL_W + tx * scale - pan;
      if (x < ROW_LABEL_W || x > W) continue;
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x, HEADER_H); ctx.lineTo(x, HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP)); ctx.stroke();
      ctx.fillText(fmtTime(tx), x, HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP) + 4);
    }

    // ── Gantt events ──
    events.forEach(ev => {
      const x = ROW_LABEL_W + ev.start * scale - pan;
      const bw = Math.max(1, ev.duration * scale);
      if (x + bw < ROW_LABEL_W || x > W) return;

      const y = HEADER_H + ev.row * (ROW_H + ROW_GAP) + 2;
      const h = ROW_H - 4;

      ctx.globalAlpha = 0.9;
      ctx.fillStyle = ev.color;
      ctx.fillRect(x, y, bw, h);
      ctx.globalAlpha = 1;

      // Label if wide enough
      if (bw > 30) {
        ctx.font = '400 9px JetBrains Mono, monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const label = ev.isComm ? ev.phase.toUpperCase() : ev.phase;
        ctx.fillText(label, x + 3, y + h / 2);
      }
    });

    ctx.restore();

    // ── Bottom separator ──
    ctx.strokeStyle = C.border;
    ctx.beginPath();
    ctx.moveTo(0, HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP));
    ctx.lineTo(W, HEADER_H + NUM_ROWS * (ROW_H + ROW_GAP));
    ctx.stroke();

    // Zoom info
    ctx.font = FONT;
    ctx.fillStyle = C.text3;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${zoom.toFixed(1)}x`, W - 12, HEADER_H / 2);

  }, [getLayout, events, zoom, totalTime, fwdEndTime, runName, layers]);

  /* ── tooltip ── */
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    const tip = tooltipRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !tip || !wrap) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const drawW = wrap.clientWidth - ROW_LABEL_W;
    const scale = (drawW * zoom) / totalTime;

    const relX = mx + panRef.current - ROW_LABEL_W;
    if (relX < 0) { tip.style.display = 'none'; return; }

    const hit = events.find(ev => {
      const x = ROW_LABEL_W + ev.start * scale - panRef.current;
      const bw = Math.max(1, ev.duration * scale);
      return mx >= x && mx <= x + bw && my >= HEADER_H + ev.row * (ROW_H + ROW_GAP) && my < HEADER_H + (ev.row + 1) * (ROW_H + ROW_GAP);
    });

    if (hit) {
      tip.innerHTML = `<b style="color:${C.text}">${hit.name}</b><br>` +
        `<span style="color:${hit.color}">\u25A0</span> ${hit.phase.toUpperCase()} · Row ${ROWS[hit.row]}<br>` +
        `<span style="color:${C.text2}">${fmtTime(hit.start)} → ${fmtTime(hit.start + hit.duration)}</span><br>` +
        `<span style="color:${C.text3}">${fmtTime(hit.duration)}</span>`;
      tip.style.display = 'block';
      tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 220) + 'px';
      tip.style.top = Math.max(8, e.clientY - 60) + 'px';
      return;
    }

    tip.style.display = 'none';
  }, [events, zoom]);

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = 'none';
  }, []);

  /* ── wheel zoom ── */
  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    setZoom(z => {
      const next = e.deltaY < 0 ? z * 1.2 : z / 1.2;
      return Math.max(0.1, Math.min(500, next));
    });
  }, []);

  /* ── pointer pan ── */
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    dragRef.current = { startX: e.clientX, startPan: panRef.current };
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);
  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return;
    panRef.current = dragRef.current.startPan - (e.clientX - dragRef.current.startX);
    draw();
  }, [draw, isDragging]);
  const handlePointerUp = useCallback(() => setIsDragging(false), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onCancel = () => setIsDragging(false);
    canvas.addEventListener('pointercancel', onCancel);
    return () => canvas.removeEventListener('pointercancel', onCancel);
  }, []);

  const zoomIn  = () => setZoom(z => Math.min(500, z * 1.4));
  const zoomOut = () => setZoom(z => Math.max(0.1, z / 1.4));
  const zoomFit = () => { panRef.current = 0; setZoom(1); };

  /* ── effects ── */
  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [draw]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  if (!layers || layers.length === 0) return null;

  const legendItems = [
    { label: 'FWD Compute', color: C.compute_fwd },
    { label: 'BWD Compute', color: C.compute_bwd },
    { label: 'TP Comm', color: C.tp },
    { label: 'DP Comm', color: C.dp },
    { label: 'EP Comm', color: C.ep },
    { label: 'DP_EP Comm', color: C.dp_ep },
  ];

  return (
    <div style={{ background: C.bg, borderRadius: 8, border: `1px solid ${C.border}`, overflow: 'hidden', fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      {/* Toolbar */}
      <div style={{ padding: '12px 20px 10px', borderBottom: `1px solid ${C.border}`, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {([['Zoom In', zoomIn], ['Zoom Out', zoomOut], ['Fit', zoomFit]] as [string, () => void][]).map(([label, fn]) => (
            <button key={label} onClick={fn}
              style={{ fontFamily: FONT, fontSize: 11, padding: '3px 10px', background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 4, color: C.text, cursor: 'pointer' }}>
              {label}
            </button>
          ))}
          <span style={{ fontFamily: FONT, fontSize: 11, color: C.text2, padding: '3px 6px' }}>{zoom.toFixed(1)}x</span>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          {legendItems.map(it => (
            <div key={it.label} style={{ display: 'flex', gap: 4, alignItems: 'center', fontSize: 10, color: C.text2, fontFamily: FONT }}>
              <div style={{ width: 8, height: 8, borderRadius: 1, background: it.color }} />
              {it.label}
            </div>
          ))}
        </div>

        <div style={{ marginLeft: 'auto', fontSize: 10, color: C.text3, fontFamily: FONT }}>
          Scroll to zoom · Drag to pan
        </div>
      </div>

      {/* Canvas */}
      <div ref={wrapRef} style={{ position: 'relative', cursor: isDragging ? 'grabbing' : 'grab' }}>
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={(e) => { handlePointerMove(e); handleMouseMove(e); }}
          onPointerUp={handlePointerUp}
          onMouseLeave={handleMouseLeave}
        />
      </div>

      {/* Tooltip */}
      <div ref={tooltipRef} style={{
        display: 'none', position: 'fixed', background: '#1e293b', border: '1px solid #475569',
        borderRadius: 6, padding: '8px 12px', fontFamily: FONT, fontSize: 11, color: C.text,
        zIndex: 9999, pointerEvents: 'none', lineHeight: 1.7,
        boxShadow: '0 8px 24px rgba(0,0,0,.5)',
      }} />
    </div>
  );
}
