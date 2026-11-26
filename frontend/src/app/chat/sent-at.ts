const FORMAT_OPTIONS: Intl.DateTimeFormatOptions = { dateStyle: 'medium', timeStyle: 'short' };

/** Browser locale and local time zone. Built on first use and reused for every message. */
let defaultFormatter: Intl.DateTimeFormat | undefined;

/**
 * Formats a message's `sent_at` (ISO-8601, as the chat API returns it) as a date and time for
 * display. Without `locales` and `timeZone` it follows the browser's locale and time zone.
 * Returns an empty string when there is no value or it is not a date, so nothing is shown.
 */
export function formatSentAt(
  sentAt: string | null | undefined,
  locales?: Intl.LocalesArgument,
  timeZone?: string,
): string {
  if (!sentAt) {
    return '';
  }

  // Python writes microseconds; the ECMAScript date format only promises milliseconds.
  const date = new Date(sentAt.replace(/(:\d{2}\.\d{3})\d+/, '$1'));
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  if (locales === undefined && timeZone === undefined) {
    defaultFormatter ??= new Intl.DateTimeFormat(undefined, FORMAT_OPTIONS);
    return defaultFormatter.format(date);
  }
  return new Intl.DateTimeFormat(locales, { ...FORMAT_OPTIONS, timeZone }).format(date);
}
