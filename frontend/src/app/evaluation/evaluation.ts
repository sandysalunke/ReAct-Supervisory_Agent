import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { Api, EvaluationView, SavedReport, errorMessage } from '../api';
import { ChartView } from '../chart';

@Component({
  selector: 'app-evaluation',
  imports: [ChartView, DecimalPipe, PercentPipe],
  templateUrl: './evaluation.html',
  styleUrl: './evaluation.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Evaluation {
  private readonly api = inject(Api);

  protected readonly reports = signal<SavedReport[] | null>(null);
  protected readonly selectedName = signal<string | null>(null);
  /** Name of an uploaded report; while set it takes precedence over saved reports. */
  protected readonly uploadedName = signal<string | null>(null);
  protected readonly view = signal<EvaluationView | null>(null);
  protected readonly loading = signal(true);
  protected readonly error = signal<string | null>(null);

  protected readonly intentLabels = computed(
    () => this.view()?.per_intent.map((row) => row.intent) ?? [],
  );
  protected readonly intentSeries = computed(() => {
    const rows = this.view()?.per_intent ?? [];
    return [
      { label: 'precision', data: rows.map((row) => row.precision) },
      { label: 'recall', data: rows.map((row) => row.recall) },
      { label: 'f1', data: rows.map((row) => row.f1) },
    ];
  });

  constructor() {
    this.api.listReports().subscribe({
      next: ({ reports }) => {
        this.reports.set(reports);
        if (reports.length > 0) {
          this.selectReport(reports[0].name);
        } else {
          this.loading.set(false);
        }
      },
      error: (error: unknown) => this.fail(error),
    });
  }

  protected selectReport(name: string): void {
    this.selectedName.set(name);
    this.load(this.api.getReport(name));
  }

  protected onUpload(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    this.uploadedName.set(file.name);
    this.load(this.api.uploadReport(file));
    input.value = '';
  }

  protected clearUpload(): void {
    this.uploadedName.set(null);
    const name = this.selectedName();
    if (name) {
      this.selectReport(name);
    } else {
      this.view.set(null);
      this.error.set(null);
    }
  }

  private load(request: ReturnType<Api['getReport']>): void {
    this.loading.set(true);
    this.error.set(null);
    this.view.set(null);
    request.subscribe({
      next: (view) => {
        this.view.set(view);
        this.loading.set(false);
      },
      error: (error: unknown) => this.fail(error),
    });
  }

  private fail(error: unknown): void {
    this.error.set(errorMessage(error));
    this.loading.set(false);
  }
}
