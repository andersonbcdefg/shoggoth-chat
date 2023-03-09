import os
import openai
import fire
import random
import tiktoken
from gpt_utils import get_reply, summarize_with_gpt_turbo
from chat_utils import Chat

def main():
    api_key = input("Enter your OpenAI API key: ")
    openai.api_key = api_key
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    system_message = input('Enter a prompt for the chatbot, e.g. "You are a helpful assistant named Karl who talks like a pirate."\n')
    assistant_name = input('Enter the name of the chatbot: ')
    chat = Chat(
        tokenizer,
        system_message=system_message,
        reply_fn=get_reply,
        summarize_fn=summarize_with_gpt_turbo,
        stop_sequence="STOP",
        assistant_name=assistant_name
    )
    while chat.take_turn():
        pass
    filename = f"{assistant_name}_chat_{random.randint(100000, 1000000)}.json"
    chat.conversation.to_json(filename)
    print("Chat saved to " + filename)


if __name__ == '__main__':
    fire.Fire(main)