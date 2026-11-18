import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  afterRenderEffect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { marked } from 'marked';
import { Api, ChatMessage, errorMessage } from '../api';

const SESSION_KEY = 'copilot.sessionId';

const ROLE_LABELS: Record<string, string> = {
  user: 'You',
  assistant: 'Copilot',
  tool: 'Tool',
  system: 'System',
};

@Component({
  selector: 'app-chat',
  templateUrl: './chat.html',
  styleUrl: './chat.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Chat {
  private readonly api = inject(Api);

  protected readonly messages = signal<ChatMessage[]>([]);
  protected readonly draft = signal('');
  protected readonly pending = signal(false);
  protected readonly error = signal<string | null>(null);
  /** File picked for the next message. */
  protected readonly selectedFile = signal<File | null>(null);
  /** File the server remembers for this session from an earlier message. */
  protected readonly rememberedFileName = signal('');

  private sessionId = this.readSessionId();
  private readonly scroller = viewChild.required<ElementRef<HTMLElement>>('scroller');
  private readonly fileInput = viewChild.required<ElementRef<HTMLInputElement>>('fileInput');

  constructor() {
    if (this.sessionId) {
      this.api.getSession(this.sessionId).subscribe({
        next: (session) => {
          this.messages.set(session.messages);
          this.rememberedFileName.set(session.uploaded_file_name);
        },
        error: (error: unknown) => {
          if (error instanceof HttpErrorResponse && error.status === 404) {
            // The backend restarted and dropped its in-memory sessions.
            this.storeSessionId(null);
          } else {
            this.error.set(errorMessage(error));
          }
        },
      });
    }

    // Keep the newest message in view.
    afterRenderEffect(() => {
      this.messages();
      this.pending();
      const element = this.scroller().nativeElement;
      element.scrollTop = element.scrollHeight;
    });
  }

  protected roleLabel(role: string): string {
    return ROLE_LABELS[role] ?? role;
  }

  protected markdown(text: string): string {
    // Angular sanitizes the HTML when it is bound to [innerHTML].
    return marked.parse(text, { async: false, breaks: true });
  }

  protected onDraftKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      this.send();
    }
  }

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile.set(input.files?.[0] ?? null);
  }

  protected clearSelectedFile(): void {
    this.selectedFile.set(null);
    this.fileInput().nativeElement.value = '';
  }

  protected send(): void {
    const prompt = this.draft().trim();
    if (!prompt || this.pending()) {
      return;
    }

    const file = this.selectedFile();
    const previousMessages = this.messages();
    this.messages.set([
      ...previousMessages,
      { role: 'user', segments: [{ type: 'text', text: prompt }] },
    ]);
    this.draft.set('');
    this.error.set(null);
    this.pending.set(true);

    this.api.sendMessage(prompt, this.sessionId, file).subscribe({
      next: (response) => {
        this.storeSessionId(response.session_id);
        // The response repeats the user's message, so replace the optimistic copy.
        this.messages.set([...previousMessages, ...response.messages]);
        this.rememberedFileName.set(response.uploaded_file_name);
        this.clearSelectedFile();
        this.pending.set(false);
      },
      error: (error: unknown) => {
        this.messages.set(previousMessages);
        this.draft.set(prompt);
        this.error.set(errorMessage(error));
        this.pending.set(false);
      },
    });
  }

  protected newChat(): void {
    if (this.sessionId) {
      this.api.deleteSession(this.sessionId).subscribe({ error: () => undefined });
    }
    this.storeSessionId(null);
    this.messages.set([]);
    this.rememberedFileName.set('');
    this.error.set(null);
    this.clearSelectedFile();
  }

  private readSessionId(): string | null {
    try {
      return sessionStorage.getItem(SESSION_KEY);
    } catch {
      return null;
    }
  }

  private storeSessionId(sessionId: string | null): void {
    this.sessionId = sessionId;
    try {
      if (sessionId) {
        sessionStorage.setItem(SESSION_KEY, sessionId);
      } else {
        sessionStorage.removeItem(SESSION_KEY);
      }
    } catch {
      // Storage can be unavailable (private mode); the chat still works for this page load.
    }
  }
}
