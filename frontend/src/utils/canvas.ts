import type { DetectionResult } from '../types/detection';

const COLORS = [
  [255, 76, 76],   // Red
  [76, 175, 80],   // Green
  [33, 150, 243],  // Blue
  [255, 193, 7],   // Amber
  [156, 39, 176],  // Purple
];

export function drawDetections(
  ctx: CanvasRenderingContext2D,
  detections: DetectionResult[],
  scaleX: number = 1,
  scaleY: number = 1,
) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    const color = COLORS[det.class_id % COLORS.length];
    const rgbStr = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

    // Semi-transparent fill
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = rgbStr;
    ctx.fillRect(sx1, sy1, sw, sh);

    // Border
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = rgbStr;
    ctx.lineWidth = 2;
    ctx.strokeRect(sx1, sy1, sw, sh);

    // Label
    const label = `${det.class_name} ${Math.round(det.score * 100)}%`;
    ctx.font = '14px Arial';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = rgbStr;
    ctx.fillRect(sx1, sy1 - 22, textWidth + 8, 22);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, sx1 + 4, sy1 - 6);
  }
}
