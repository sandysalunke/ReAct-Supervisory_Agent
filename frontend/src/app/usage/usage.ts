import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Api, UsageSummary, errorMessage } from '../api';
import { ChartView } from '../chart';

@Component({
  selector: 'app-usage',
  imports: [ChartView, DecimalPipe],
  templateUrl: './usage.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Usage {
  protected readonly usage = signal<UsageSummary | null>(null);
  protected readonly error = signal<string | null>(null);

  protected readonly agentLabels = computed(
    () => this.usage()?.tokens_by_agent.map((row) => row.agent) ?? [],
  );
  protected readonly agentTokens = computed(() => [
    {
      label: 'Total tokens',
      data: this.usage()?.tokens_by_agent.map((row) => row.total_tokens) ?? [],
    },
  ]);
  protected readonly modelLabels = computed(
    () => this.usage()?.tokens_by_model.map((row) => row.model) ?? [],
  );
  protected readonly modelTokens = computed(() => [
    {
      label: 'Total tokens',
      data: this.usage()?.tokens_by_model.map((row) => row.total_tokens) ?? [],
    },
  ]);
  protected readonly callLabels = computed(
    () => this.usage()?.calls_by_agent.map((row) => row.agent) ?? [],
  );
  protected readonly agentCalls = computed(() => [
    { label: 'Calls', data: this.usage()?.calls_by_agent.map((row) => row.calls) ?? [] },
  ]);
  protected readonly dayLabels = computed(
    () => this.usage()?.daily_tokens.map((row) => row.date) ?? [],
  );
  protected readonly dailyTokens = computed(() => [
    {
      label: 'Total tokens',
      data: this.usage()?.daily_tokens.map((row) => row.total_tokens) ?? [],
    },
  ]);

  constructor() {
    inject(Api)
      .getUsage()
      .subscribe({
        next: (usage) => this.usage.set(usage),
        error: (error: unknown) => this.error.set(errorMessage(error)),
      });
  }
}
