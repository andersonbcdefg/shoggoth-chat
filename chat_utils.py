import json

class ChatHistory:
  def __init__(self, tokenizer, system_message=None, initial_messages=None, initial_summary=None):
    self.tokenizer = tokenizer
    self.system_message = system_message
    self.summary = initial_summary
    self.messages = [] if initial_messages is None else initial_messages
    self.archive = []

  def __getitem__(self, idx):
    return self.messages[idx]

  def __str__(self):
    result = ""
    if self.summary is not None:
      result += "Summary of the previous conversation: " + self.summary
      result += "\n\n" 
    result += "\n\n".join(__class__.message_to_string(m) for m in self.messages)
    return result

  def __len__(self):
    return len(self.messages)

  def to_json(self, file):
    with open(file, "w+") as f:
      json.dump({
          "system_message": self.system_message,
          "summary": self.summary,
          "messages": self.messages,
          "archive": self.archive
      }, f)

  def load_json(self, file):
    with open(file) as f:
      chat_history = json.load(f)
      self.summary = chat_history["summary"]
      self.messages = chat_history["messages"]
      self.archive = chat_history["archive"]

  def add_user_message(self, content):
    new_msg = { "role": "user", "content": content}
    self.messages.append(new_msg)

  def add_assistant_message(self, content):
    new_msg = { "role": "assistant", "content": content }
    self.messages.append(new_msg)
  
  @staticmethod
  def message_to_string(message):
    result = message["role"].capitalize() + ": "
    result += message["content"]
    return result

  # Flushes messages out of the conversation buffer and into the summary to make room for more messages.
  # Requires a summarize_fn to call to create new summaries, which must accept (instructions, content).
  def flush_to_summary(self, max_buffer_tokens, summarize_fn):
    messages_to_summarize = []
    while self._count_message_tokens(self.messages) > max_buffer_tokens:
      popped = self.messages.pop(0)
      self.archive.append(popped)
      messages_to_summarize.append(popped)
    transcript = "\n\n".join(__class__.message_to_string(message) for message in messages_to_summarize)
    summarize_instructions = "Summarize the following conversation between the user and the assistant."
    new_summary = summarize_fn(summarize_instructions, transcript)
    # If necessary, combine new and old summaries.
    if self.summary is not None:
      summarize_instructions = "Summarize the following text into a single paragraph."
      content = self.summary + "\n\n" + new_summary
      new_summary = summarize_fn(summarize_instructions, content)
    self.summary = new_summary

  def total_message_tokens(self):
    return self._count_message_tokens(self.messages)

  # Messages accepted as argument because you may want to count a subset
  def _count_message_tokens(self, messages):
    num_tokens = 0
    for message in messages:
      num_tokens += 4
      for key, value in message.items():
        num_tokens += len(self.tokenizer.encode(value))
        if key == "name":  # if there's a name, the role is omitted
          num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

  def get_messages_with_system_prompts(self):
    result = []
    initial_prompt = ""
    if self.system_message is not None:
      initial_prompt += self.system_message
    if self.summary is not None:
      initial_prompt += "\n\nHere is a summary of your previous conversation with the user: "
      initial_prompt += self.summary
    if len(initial_prompt) > 0:
      result.append({
          "role": "system",
          "content": initial_prompt
      })
    result.extend(self.messages)
    return result

class Chat:
  def __init__(self, tokenizer, system_message, reply_fn, summarize_fn, stop_sequence,
               assistant_name="Assistant", max_buffer_tokens=3000, reset_buffer_len=1200, debug=False):
    self.debug = debug
    self.assistant_name = assistant_name
    self.stop_sequence = stop_sequence
    self.max_buffer_tokens = max_buffer_tokens
    self.reset_buffer_len = reset_buffer_len
    self.conversation = ChatHistory(tokenizer, system_message)
    self.reply_fn = reply_fn
    self.summarize_fn = summarize_fn
  
  def take_turn(self):
    user_message = input("User: ")
    if user_message == self.stop_sequence:
      return False
    self.conversation.add_user_message(user_message)
    reply = self.reply_fn(self.conversation)
    self.conversation.add_assistant_message(reply)
    print(f"{self.assistant_name}: " + reply)

    # Offload to summary if necessary
    if self.conversation.total_message_tokens() > self.max_buffer_tokens:
      if self.debug: print("Memory full! Flushing to summary...")
      self.conversation.flush_to_summary(self.reset_buffer_len, self.summarize_fn)
    return True

  def chat(self):
    print(f"You are now chatting with {self.assistant_name}. Type '{self.stop_sequence}' to end the conversation.")
    while True:
      result = self.take_turn()
      if not result:
        break