'use strict';
document.querySelectorAll('form.quiz').forEach(form => {
  form.addEventListener('submit', event => {
    event.preventDefault();
    const selected = form.querySelector('input:checked');
    const feedback = form.querySelector('.feedback');
    if (!selected) {
      feedback.textContent = 'Choose an answer first. Try recalling it before checking the lesson.';
      feedback.className = 'feedback retry';
      return;
    }
    const correct = selected.value === form.dataset.answer;
    feedback.textContent = correct ? form.dataset.correct : form.dataset.retry;
    feedback.className = `feedback ${correct ? 'correct' : 'retry'}`;
  });
});
