import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { Api, errorMessage } from '../api';

@Component({
  selector: 'app-mcp',
  templateUrl: './mcp.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Mcp {
  private readonly api = inject(Api);

  protected readonly pending = signal(false);
  protected readonly result = signal<string | null>(null);
  protected readonly error = signal<string | null>(null);

  protected run(): void {
    this.pending.set(true);
    this.result.set(null);
    this.error.set(null);
    this.api.runMcpAddNumbers().subscribe({
      next: (response) => {
        this.result.set(String(response.result));
        this.pending.set(false);
      },
      error: (error: unknown) => {
        this.error.set(errorMessage(error));
        this.pending.set(false);
      },
    });
  }
}
