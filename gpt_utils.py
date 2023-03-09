import openai

def get_reply(conversation):
  messages = conversation.get_messages_with_system_prompts()
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=messages
  )
  return response['choices'][0]['message']['content']

def summarize_with_gpt_turbo(instructions, content):
  messages = [
      {"role": "user", "content": instructions + "\n\n" + content}
  ]
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=messages
  )
  return response['choices'][0]['message']['content']