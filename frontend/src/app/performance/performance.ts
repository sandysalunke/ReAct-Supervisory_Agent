import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { Api, PerformanceSummary, QualityScores, errorMessage } from '../api';
import { ChartView } from '../chart';

const METRICS: { key: keyof QualityScores; label: string }[] = [
  { key: 'groundedness', label: 'Groundedness' },
  { key: 'relevance', label: 'Relevance' },
  { key: 'completeness', label: 'Completeness' },
];

@Component({
  selector: 'app-performance',
  imports: [ChartView, DatePipe, DecimalPipe],
  templateUrl: './performance.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Performance {
  protected readonly metrics = METRICS;
  protected readonly performance = signal<PerformanceSummary | null>(null);
  protected readonly error = signal<string | null>(null);

  protected readonly metricLabels = METRICS.map((metric) => metric.label);
  protected readonly averageSeries = computed(() => [
    {
      label: 'Score',
      data: METRICS.map((metric) => this.performance()?.averages[metric.key] ?? null),
    },
  ]);
  protected readonly dayLabels = computed(
    () => this.performance()?.daily.map((day) => day.date) ?? [],
  );
  protected readonly dailySeries = computed(() =>
    METRICS.map((metric) => ({
      label: metric.label,
      data: this.performance()?.daily.map((day) => day[metric.key]) ?? [],
    })),
  );

  constructor() {
    inject(Api)
      .getPerformance()
      .subscribe({
        next: (performance) => this.performance.set(performance),
        error: (error: unknown) => this.error.set(errorMessage(error)),
      });
  }
}
