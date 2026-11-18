import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

export type MessageSegment = { type: 'text'; text: string } | { type: 'image'; url: string };

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool' | 'system' | string;
  segments: MessageSegment[];
  interrupted?: boolean;
}

export interface ChatResponse {
  session_id: string;
  thread_id: string;
  workflow_status: 'COMPLETED' | 'INTERRUPTED';
  messages: ChatMessage[];
  uploaded_file_name: string;
}

export interface ChatSession {
  session_id: string;
  messages: ChatMessage[];
  uploaded_file_name: string;
}

export interface UsageSummary {
  total_calls: number;
  total_tokens: number;
  avg_tokens_per_call: number;
  tokens_by_agent: { agent: string; total_tokens: number; percent: number }[];
  calls_by_agent: { agent: string; calls: number }[];
  tokens_by_model: { model: string; total_tokens: number }[];
  daily_tokens: { date: string; total_tokens: number }[];
  top_threads: { thread_id: string; total_tokens: number }[];
}

export interface QualityScores {
  groundedness: number | null;
  relevance: number | null;
  completeness: number | null;
}

export interface PerformanceSummary {
  rated_responses: number;
  averages: QualityScores;
  daily: ({ date: string } & QualityScores)[];
  low_scores: ({ timestamp: string | null; thread_id: string | null } & QualityScores)[];
  low_score_threshold: number;
}

export interface SavedReport {
  name: string;
  path: string;
}

export interface EvaluationView {
  source: string;
  summary: {
    sample_count: number;
    exact_matches: number;
    exact_match_accuracy: number;
    intent_micro_precision: number;
    intent_micro_recall: number;
    intent_micro_f1: number;
    invalid_plan_count: number;
  };
  per_intent: { intent: string; precision: number; recall: number; f1: number }[];
  mismatches: {
    id: string;
    prompt: string;
    expected_intents: string[];
    predicted_intents: string[];
    error: string;
  }[];
  warning: string | null;
}

/** Turns a failed request into a message that can be shown to the user. */
export function errorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    const detail = error.error?.detail;
    if (typeof detail === 'string') {
      return detail;
    }
    // The dev-server proxy answers with a bare 5xx when the backend is down.
    if (error.status === 0 || error.status >= 500) {
      return 'Could not reach the API. Start the backend with "python main.py".';
    }
    return `Request failed (${error.status} ${error.statusText}).`;
  }
  return error instanceof Error ? error.message : String(error);
}

@Injectable({ providedIn: 'root' })
export class Api {
  private readonly http = inject(HttpClient);

  sendMessage(message: string, sessionId: string | null, file: File | null) {
    const form = new FormData();
    form.append('message', message);
    if (sessionId) {
      form.append('session_id', sessionId);
    }
    if (file) {
      form.append('file', file, file.name);
    }
    return this.http.post<ChatResponse>('/api/chat', form);
  }

  getSession(sessionId: string) {
    return this.http.get<ChatSession>(`/api/chat/${encodeURIComponent(sessionId)}`);
  }

  deleteSession(sessionId: string) {
    return this.http.delete<void>(`/api/chat/${encodeURIComponent(sessionId)}`);
  }

  getUsage() {
    return this.http.get<UsageSummary>('/api/usage');
  }

  getPerformance() {
    return this.http.get<PerformanceSummary>('/api/performance');
  }

  listReports() {
    return this.http.get<{ reports: SavedReport[] }>('/api/evaluation/reports');
  }

  getReport(name: string) {
    return this.http.get<EvaluationView>(`/api/evaluation/reports/${encodeURIComponent(name)}`);
  }

  uploadReport(file: File): Observable<EvaluationView> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<EvaluationView>('/api/evaluation/reports', form);
  }

  runMcpAddNumbers() {
    return this.http.post<{ result: string | null }>('/api/mcp/add-numbers', null);
  }
}
