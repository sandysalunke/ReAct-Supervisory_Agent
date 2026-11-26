import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ChatMessage, ChatResponse, ChatSession } from '../api';
import { Chat } from './chat';
import { formatSentAt } from './sent-at';

const SESSION_KEY = 'copilot.sessionId';
const SESSION_ID = 'session-1';

const USER_MESSAGE: ChatMessage = {
  role: 'user',
  segments: [{ type: 'text', text: 'Draw the sales chart' }],
  interrupted: false,
  sent_at: '2026-05-04T09:15:30.123456+00:00',
};

const ASSISTANT_MESSAGE: ChatMessage = {
  role: 'assistant',
  segments: [
    { type: 'text', text: 'Here is the **chart**:' },
    { type: 'image', url: '/api/files/images/chart.png' },
  ],
  sent_at: '2026-05-04T09:15:42.654321+00:00',
};

const SYSTEM_MESSAGE: ChatMessage = {
  role: 'system',
  segments: [{ type: 'text', text: 'Should I continue?' }],
  interrupted: true,
  sent_at: '2026-05-04T09:16:05.000001+00:00',
};

describe('Chat', () => {
  let fixture: ComponentFixture<Chat>;
  let http: HttpTestingController;

  beforeEach(() => {
    sessionStorage.clear();
    TestBed.configureTestingModule({
      imports: [Chat],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
    sessionStorage.clear();
  });

  async function settle(): Promise<void> {
    fixture.detectChanges();
    await fixture.whenStable();
  }

  /** Opens the chat screen with a stored session and answers the history request. */
  async function loadSession(messages: ChatMessage[]): Promise<void> {
    sessionStorage.setItem(SESSION_KEY, SESSION_ID);
    fixture = TestBed.createComponent(Chat);
    await settle();
    const session: ChatSession = {
      session_id: SESSION_ID,
      messages,
      uploaded_file_name: '',
    };
    http.expectOne({ method: 'GET', url: `/api/chat/${SESSION_ID}` }).flush(session);
    await settle();
  }

  /** Types a message and submits the composer; the POST is left for the test to answer. */
  async function submit(text: string): Promise<void> {
    const root: HTMLElement = fixture.nativeElement;
    const textarea = root.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]')!;
    textarea.value = text;
    textarea.dispatchEvent(new Event('input'));
    await settle();
    root
      .querySelector<HTMLFormElement>('form.composer')!
      .dispatchEvent(new Event('submit', { cancelable: true }));
    await settle();
  }

  function articles(): HTMLElement[] {
    const root: HTMLElement = fixture.nativeElement;
    return Array.from(root.querySelectorAll<HTMLElement>('article.message'));
  }

  /** Checks that the article shows exactly one time, and that it is the message's. */
  function expectSentAt(article: HTMLElement, message: ChatMessage): void {
    const times = article.querySelectorAll('time.sent-at');
    expect(times.length).toBe(1);
    const label = formatSentAt(message.sent_at);
    expect(label).not.toBe('');
    expect(times[0].getAttribute('datetime')).toBe(message.sent_at);
    expect(times[0].textContent?.trim()).toBe(label);
  }

  it('shows the date and time of every message when a session loads, for all roles', async () => {
    const messages = [USER_MESSAGE, ASSISTANT_MESSAGE, SYSTEM_MESSAGE];
    await loadSession(messages);

    const shown = articles();
    expect(shown.length).toBe(3);
    shown.forEach((article, index) => expectSentAt(article, messages[index]));
    expect(shown.map((article) => article.querySelector('.role')?.textContent?.trim())).toEqual([
      'You',
      'Copilot',
      'System',
    ]);
  });

  it('keeps the message content next to the time', async () => {
    await loadSession([USER_MESSAGE, ASSISTANT_MESSAGE, SYSTEM_MESSAGE]);

    const [user, assistant, system] = articles();
    expect(user.querySelector('.plain')?.textContent).toBe('Draw the sales chart');
    expect(assistant.querySelector('.markdown strong')?.textContent).toBe('chart');
    expect(assistant.querySelector('img')?.getAttribute('src')).toBe(
      '/api/files/images/chart.png',
    );
    expect(system.querySelector('.markdown')?.textContent).toContain('Should I continue?');
    expect(system.classList.contains('interrupted')).toBe(true);
    // The time sits on the role line, not inside the content.
    for (const article of [user, assistant, system]) {
      expect(article.querySelector('.meta time.sent-at')).not.toBeNull();
      expect(article.querySelector('.plain time, .markdown time')).toBeNull();
    }
  });

  it('shows nothing for a message without a usable time', async () => {
    const untimed: ChatMessage = {
      role: 'assistant',
      segments: [{ type: 'text', text: 'An older reply' }],
    };
    const invalid: ChatMessage = { ...untimed, sent_at: 'not a date' };
    await loadSession([untimed, invalid, USER_MESSAGE]);

    const shown = articles();
    expect(shown.length).toBe(3);
    expect(shown[0].querySelector('time')).toBeNull();
    expect(shown[1].querySelector('time')).toBeNull();
    expect(shown[0].querySelector('.role')?.textContent?.trim()).toBe('Copilot');
    expect(shown[0].querySelector('.markdown')?.textContent).toContain('An older reply');
    expect(fixture.nativeElement.textContent).not.toContain('Invalid Date');
    expectSentAt(shown[2], USER_MESSAGE);
  });

  it('shows the time of a sent message and of its reply', async () => {
    fixture = TestBed.createComponent(Chat);
    await settle();
    await submit('Draw the sales chart');

    // Until the server answers, the message is shown without a time.
    let shown = articles();
    expect(shown.length).toBe(1);
    expect(shown[0].querySelector('.plain')?.textContent).toBe('Draw the sales chart');
    expect(shown[0].querySelector('time')).toBeNull();

    const response: ChatResponse = {
      session_id: SESSION_ID,
      thread_id: 'thread-1',
      workflow_status: 'COMPLETED',
      messages: [USER_MESSAGE, ASSISTANT_MESSAGE],
      uploaded_file_name: '',
    };
    http.expectOne({ method: 'POST', url: '/api/chat' }).flush(response);
    await settle();

    shown = articles();
    expect(shown.length).toBe(2);
    expectSentAt(shown[0], USER_MESSAGE);
    expectSentAt(shown[1], ASSISTANT_MESSAGE);
    expect(shown[1].querySelector('img')).not.toBeNull();
    expect(sessionStorage.getItem(SESSION_KEY)).toBe(SESSION_ID);
  });

  it('shows the time of the system message when the reply is an interrupt', async () => {
    await loadSession([USER_MESSAGE, ASSISTANT_MESSAGE]);
    await submit('interrupt me');

    const question: ChatMessage = {
      role: 'user',
      segments: [{ type: 'text', text: 'interrupt me' }],
      interrupted: false,
      sent_at: '2026-05-04T09:16:00.000000+00:00',
    };
    const response: ChatResponse = {
      session_id: SESSION_ID,
      thread_id: 'thread-1',
      workflow_status: 'INTERRUPTED',
      messages: [question, SYSTEM_MESSAGE],
      uploaded_file_name: '',
    };
    http.expectOne({ method: 'POST', url: '/api/chat' }).flush(response);
    await settle();

    const shown = articles();
    expect(shown.length).toBe(4);
    // The earlier messages keep their times; the new ones get theirs.
    expectSentAt(shown[0], USER_MESSAGE);
    expectSentAt(shown[1], ASSISTANT_MESSAGE);
    expectSentAt(shown[2], question);
    expectSentAt(shown[3], SYSTEM_MESSAGE);
    expect(shown[3].classList.contains('interrupted')).toBe(true);
  });

  it('still clears the conversation and deletes the session on New chat', async () => {
    await loadSession([USER_MESSAGE, ASSISTANT_MESSAGE]);
    expect(articles().length).toBe(2);

    const root: HTMLElement = fixture.nativeElement;
    root.querySelector<HTMLButtonElement>('header button')!.click();
    await settle();

    http
      .expectOne({ method: 'DELETE', url: `/api/chat/${SESSION_ID}` })
      .flush(null, { status: 204, statusText: 'No Content' });
    await settle();

    expect(articles().length).toBe(0);
    expect(root.querySelector('time')).toBeNull();
    expect(root.querySelector('.empty')).not.toBeNull();
    expect(sessionStorage.getItem(SESSION_KEY)).toBeNull();
  });
});
