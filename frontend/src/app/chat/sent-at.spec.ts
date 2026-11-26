import { formatSentAt } from './sent-at';

describe('formatSentAt', () => {
  it('formats the date and time for the given locale and time zone', () => {
    const british = formatSentAt('2026-05-04T09:15:30+00:00', 'en-GB', 'UTC');
    expect(british).toContain('4 May 2026');
    expect(british).toContain('09:15');

    const american = formatSentAt('2026-05-04T09:15:30+00:00', 'en-US', 'UTC');
    expect(american).toContain('May 4, 2026');
    expect(american).toMatch(/9:15\sAM/);
  });

  it('converts the UTC value to the requested time zone', () => {
    const tokyo = formatSentAt('2026-05-04T23:30:00+00:00', 'en-GB', 'Asia/Tokyo');
    expect(tokyo).toContain('5 May 2026');
    expect(tokyo).toContain('08:30');
  });

  it('reads the microsecond precision that Python writes', () => {
    const fromPython = formatSentAt('2026-05-04T09:15:30.123456+00:00', 'en-GB', 'UTC');
    expect(fromPython).not.toBe('');
    expect(fromPython).toBe(formatSentAt('2026-05-04T09:15:30.123+00:00', 'en-GB', 'UTC'));
    expect(fromPython).toContain('4 May 2026');
    expect(fromPython).toContain('09:15');
  });

  it('uses the browser locale and time zone by default', () => {
    const sentAt = '2026-05-04T09:15:30.123456+00:00';
    const expected = new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date('2026-05-04T09:15:30.123Z'));
    expect(formatSentAt(sentAt)).toBe(expected);
    // The cached formatter gives the same answer on later calls.
    expect(formatSentAt(sentAt)).toBe(expected);
  });

  it('returns an empty string when there is no usable value', () => {
    expect(formatSentAt(undefined)).toBe('');
    expect(formatSentAt(null)).toBe('');
    expect(formatSentAt('')).toBe('');
    expect(formatSentAt('not a date')).toBe('');
    expect(formatSentAt('not a date', 'en-GB', 'UTC')).toBe('');
  });
});
