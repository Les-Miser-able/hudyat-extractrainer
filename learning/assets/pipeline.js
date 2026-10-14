'use strict';
// An illustrative count of outputs from one accepted source, not model inference.
document.querySelectorAll('[data-source-demo]').forEach(demo => {
  function update() {
    const kind = demo.querySelector('[name="kind"]').value;
    const split = demo.querySelector('[name="split"]').value;
    const extras = Number(demo.querySelector('[name="extras"]').value);
    const isStatic = kind === 'image';
    const count = isStatic ? 8 : 1 + (split === 'train' ? extras : 0);
    const upTo = !isStatic && split === 'train' && extras > 0;
    demo.querySelector('[data-count]').textContent = `${upTo ? 'Up to ' : ''}${count} prepared ${count === 1 ? 'sample' : 'samples'}`;
    demo.querySelector('[data-shape]').textContent = 'Each sample: 32 timesteps × 128 features';
    demo.querySelector('[data-explanation]').textContent = isStatic
      ? `The pose is repeated over 32 timesteps, then jitter is added to present-hand coordinates. With the default eight variants, this also happens for static sources assigned to ${split}. All eight remain in that one split.`
      : `${kind === 'video' ? 'The full decoded clip is converted to hand landmarks, then' : 'Existing compatible landmarks skip detection and'} the full sequence is resampled to 32 timesteps. ${upTo ? 'Only training sequences can receive timing extras; duplicate or invalid candidates can reduce the count.' : split === 'train' ? 'No timing extras were requested, so this source receives one base sequence.' : 'This source receives one base sequence; timing extras do not apply to validation or test.'}`;
    demo.querySelector('[data-group]').textContent = 'Still one original source in one source group. Variants do not become independent recordings.';
  }
  demo.addEventListener('change', update);
  update();
});

// Keep the full reference available when printing collapsed details sections.
let printOpened = [];
window.addEventListener('beforeprint', () => {
  printOpened = [...document.querySelectorAll('details:not([open])')];
  printOpened.forEach(detail => { detail.open = true; });
});
window.addEventListener('afterprint', () => {
  printOpened.forEach(detail => { detail.open = false; });
  printOpened = [];
});
