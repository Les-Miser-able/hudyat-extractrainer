// Timing only: HUDYAT owns all detection, classification, and confirmation decisions.
export function createTimingTrial(target) {
  let started = null, firstFrame = null, predicted = null;
  return {
    capture(frameStarted, frame) {
      if (firstFrame === null) firstFrame = frameStarted;
      if (started === null && (frame[126] || frame[127])) started = frameStarted;
      return started;
    },
    prediction(result) { if (typeof result?.prediction === 'string') predicted = result.prediction; },
    get started() { return started; },
    get firstFrame() { return firstFrame; },
    complete(correct, paintedAt, timeout = false) {
      const origin = started ?? (timeout ? firstFrame : null);
      if (origin === null || paintedAt < origin) throw Error('No valid camera start frame was measured.');
      return {target, predicted:timeout ? null : predicted, correct:timeout ? false : correct,
        response_ms:timeout ? null : paintedAt - origin, elapsed_ms:paintedAt - origin, timeout};
    }
  };
}
