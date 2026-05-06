const hero = document.getElementById('hero');
const conversation = document.getElementById('conversation');
const bottomInput = document.getElementById('bottomInput');
const chatArea = document.getElementById('chatArea');
const userInput = document.getElementById('userInput');
const heroInput = document.getElementById('heroInput');

let availableTags = [];

// ─── Load all tags for correction dropdown ─────────────────────────────────
async function loadTags() {
  try {
    const res = await fetch('/tags');
    const data = await res.json();
    availableTags = data.tags || [];
  } catch (err) {
    console.error('Failed to load tags:', err);
  }
}
loadTags();

// ─── Listeners ─────────────────────────────────────────────────────────────
heroInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); sendFromHero(); }
});

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); sendMessage(); }
});

// ─── View transitions ──────────────────────────────────────────────────────
function switchToConversation() {
  hero.classList.add('hidden');
  conversation.classList.add('active');
  bottomInput.classList.add('active');
  setTimeout(() => userInput.focus(), 50);
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

// ─── Render messages ───────────────────────────────────────────────────────
function appendMessage(text, sender, botData = null) {
  const msg = document.createElement('div');
  msg.classList.add('message', sender);

  const avatar = document.createElement('div');
  avatar.classList.add('avatar', sender === 'bot' ? 'bot-avatar' : 'user-avatar');
  avatar.textContent = sender === 'bot' ? '🌿' : '👤';

  const wrapper = document.createElement('div');
  wrapper.classList.add('bubble-wrapper');

  const bubble = document.createElement('div');
  bubble.classList.add('bubble');
  bubble.textContent = text;

  wrapper.appendChild(bubble);

  if (sender === 'bot' && botData && botData.user_message) {
    wrapper.appendChild(createFeedbackButtons(botData));
  }

  msg.appendChild(avatar);
  msg.appendChild(wrapper);
  chatArea.appendChild(msg);
  scrollToBottom();
}

// ─── Feedback UI ───────────────────────────────────────────────────────────
function createFeedbackButtons(botData) {
  const fb = document.createElement('div');
  fb.classList.add('feedback');

  const helpfulBtn = document.createElement('button');
  helpfulBtn.classList.add('fb-btn', 'fb-helpful');
  helpfulBtn.innerHTML = '👍 Helpful';
  helpfulBtn.onclick = () => handleHelpful(fb, botData);

  const notRightBtn = document.createElement('button');
  notRightBtn.classList.add('fb-btn', 'fb-wrong');
  notRightBtn.innerHTML = '👎 Not quite right';
  notRightBtn.onclick = () => handleNotRight(fb, botData);

  fb.appendChild(helpfulBtn);
  fb.appendChild(notRightBtn);
  return fb;
}

async function handleHelpful(fbContainer, botData) {
  fbContainer.innerHTML = '<span class="fb-thanks">✓ Thanks for the feedback</span>';
  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_message: botData.user_message,
        tag: botData.tag
      })
    });
  } catch (err) {
    console.error('Feedback error:', err);
  }
}

function handleNotRight(fbContainer, botData) {
  fbContainer.innerHTML = '';

  const form = document.createElement('div');
  form.classList.add('correction-form');

  const label = document.createElement('div');
  label.classList.add('correction-label');
  label.textContent = 'What category should this have been?';
  form.appendChild(label);

  const select = document.createElement('select');
  select.classList.add('correction-select');
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '— Select a category —';
  placeholder.disabled = true;
  placeholder.selected = true;
  select.appendChild(placeholder);

  availableTags.forEach(tag => {
    const opt = document.createElement('option');
    opt.value = tag;
    opt.textContent = tag;
    select.appendChild(opt);
  });

  const newTagOpt = document.createElement('option');
  newTagOpt.value = '__new__';
  newTagOpt.textContent = '+ Create a new category';
  select.appendChild(newTagOpt);
  form.appendChild(select);

  const newTagInput = document.createElement('input');
  newTagInput.type = 'text';
  newTagInput.classList.add('correction-input');
  newTagInput.placeholder = 'New category name (e.g. kaudulla_animals)';
  newTagInput.style.display = 'none';
  form.appendChild(newTagInput);

  const newResponseInput = document.createElement('textarea');
  newResponseInput.classList.add('correction-input');
  newResponseInput.placeholder = 'What should the correct response be? (optional but recommended)';
  newResponseInput.style.display = 'none';
  newResponseInput.rows = 2;
  form.appendChild(newResponseInput);

  select.onchange = () => {
    const isNew = select.value === '__new__';
    newTagInput.style.display = isNew ? 'block' : 'none';
    // Show response input for ALL selections so the bot knows what to say
    newResponseInput.style.display = select.value ? 'block' : 'none';
  };

  const submitBtn = document.createElement('button');
  submitBtn.classList.add('correction-submit');
  submitBtn.textContent = 'Save correction';
  submitBtn.onclick = async () => {
    let chosenTag = select.value;
    let newResponse = '';

    if (!chosenTag) { alert('Please pick a category.'); return; }

    if (chosenTag === '__new__') {
      chosenTag = newTagInput.value.trim().replace(/\s+/g, '_').toLowerCase();
      newResponse = newResponseInput.value.trim();
      if (!chosenTag) { alert('Please enter a name for the new category.'); return; }
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Saving...';

    try {
      const res = await fetch('/correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_message: botData.user_message,
          correct_tag: chosenTag,
          new_response: newResponse
        })
      });
      const data = await res.json();

      if (data.retrained) {
        fbContainer.innerHTML = '<span class="fb-thanks">🎉 Correction saved — model retrained, I just learned new things!</span>';
        loadTags();
      } else {
        fbContainer.innerHTML = `<span class="fb-thanks">✓ Saved — ${data.corrections_until_retrain} more until I retrain</span>`;
      }
    } catch (err) {
      console.error('Correction error:', err);
      fbContainer.innerHTML = '<span class="fb-error">Could not save correction. Try again.</span>';
    }
  };
  form.appendChild(submitBtn);

  fbContainer.appendChild(form);
  scrollToBottom();
}

// ─── Typing indicator ─────────────────────────────────────────────────────
function showTyping() {
  const typing = document.createElement('div');
  typing.classList.add('typing');
  typing.id = 'typingIndicator';
  typing.innerHTML = `
    <div class="avatar bot-avatar">🌿</div>
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>`;
  chatArea.appendChild(typing);
  scrollToBottom();
}

function removeTyping() {
  const t = document.getElementById('typingIndicator');
  if (t) t.remove();
}

// ─── Sending ──────────────────────────────────────────────────────────────
async function sendToBackend(text) {
  appendMessage(text, 'user');
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    removeTyping();
    appendMessage(data.response, 'bot', {
      tag: data.tag,
      user_message: data.user_message
    });
  } catch (err) {
    removeTyping();
    appendMessage("Sorry, something went wrong. Please try again.", 'bot');
  }
}

function sendFromHero() {
  const text = heroInput.value.trim();
  if (!text) return;
  heroInput.value = '';
  switchToConversation();
  sendToBackend(text);
}

function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;
  userInput.value = '';
  sendToBackend(text);
}

function sendSuggestion(text) {
  switchToConversation();
  sendToBackend(text);
}