import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  effect,
  inject,
  input,
  viewChild,
} from '@angular/core';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

export interface ChartSeries {
  label: string;
  data: (number | null)[];
}

/** Bar or line chart themed from the app's CSS variables. */
@Component({
  selector: 'app-chart',
  template: `<canvas #canvas role="img" [attr.aria-label]="label()"></canvas>`,
  styles: `
    :host {
      display: block;
      position: relative;
      height: 100%;
    }
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChartView {
  readonly type = input<'bar' | 'line'>('bar');
  readonly labels = input.required<string[]>();
  readonly series = input.required<ChartSeries[]>();
  readonly label = input('Chart');
  /** Upper bound of the value axis, e.g. 100 for percentages. */
  readonly max = input<number | undefined>(undefined);

  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private chart: Chart | undefined;

  constructor() {
    effect(() => {
      const canvas = this.canvas().nativeElement;
      const styles = getComputedStyle(canvas);
      const cssVar = (name: string) => styles.getPropertyValue(name).trim();
      const colors = [cssVar('--series-1'), cssVar('--series-2'), cssVar('--series-3')];
      const series = this.series();

      this.chart?.destroy();
      this.chart = new Chart(canvas, {
        type: this.type(),
        data: {
          labels: this.labels(),
          datasets: series.map((entry, index) => ({
            label: entry.label,
            data: entry.data,
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length],
            borderWidth: this.type() === 'line' ? 2 : 0,
            borderRadius: 4,
            pointRadius: 3,
            spanGaps: true,
          })),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: {
              display: series.length > 1,
              labels: { color: cssVar('--text-muted'), boxWidth: 12 },
            },
          },
          scales: {
            x: {
              ticks: { color: cssVar('--text-muted') },
              grid: { display: false },
              border: { color: cssVar('--border') },
            },
            y: {
              beginAtZero: true,
              max: this.max(),
              ticks: { color: cssVar('--text-muted') },
              grid: { color: cssVar('--border') },
              border: { display: false },
            },
          },
        },
      });
    });

    inject(DestroyRef).onDestroy(() => this.chart?.destroy());
  }
}
