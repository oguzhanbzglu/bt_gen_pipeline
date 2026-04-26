import json
from typing import TypedDict

from imap_tools import MailBox, AND

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END


IMAP_HOST="imap.gmail.com"
IMAP_USER="email@gmail.com"
IMAP_PASSWORD="pass"
IMAP_FOLDER = 'INBOX'

CHAT_MODEL = 'llama3.1:8b'

class ChatState(TypedDict):
    messages: list


def connect():
    mail_box = MailBox(IMAP_HOST)
    mail_box.login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)

    return mail_box

@tool
def list_unread_emails():
    """
    Return a bullet list of every UNREAD message's UID, subject, date and sender.
    """
    print("List Unread Emails Tool Called")

    with connect() as mb:
        unread = list(mb.fetch(criteria = AND(seen=False), headers_only=True, mark_seen=False))

    if not unread:
        return 'You have no unread messages.'

    response = json.dumps([
        {
            'uid': mail.uid,
            'date': mail.date.astimezone().strftime('%Y-%m-%d %H:%M'),
            'subject': mail.subject,
            'sender': mail.from_
        } for mail in unread
    ])

    return response

@tool
def summarize_email(uid):
    """
    Summarize an email given it's IMAP UID. Return a short summary of the emails content / body in plan text.
    """
    print("Summarize E-Mail Tool Called on ", uid)

    with connect() as mb:
        mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)

        if not mail:
            return f"Could not summarize email with UID {uid}"

        prompt = (
            "Summarize this email for concisely:\n\n"
            f"Subject: {mail.subject}\n\n"
            f"Sender: {mail.from_}\n\n"
            f"Date: {mail.data}\n\n"
            f"{mail.text or mail.html}"
        )

        return raw_llm.invoke(prompt).content

# LangGraph part
llm = init_chat_model(CHAT_MODEL, model_provider='ollama')
llm = llm.bind_tools([list_unread_emails, summarize_email])

raw_llm = init_chat_model(CHAT_MODEL, model_provider='ollama')


def llm_node(state):
    response = llm.invoke(state['messages'])
    return {'messages': state['messages'] + [response]}

def router(state):
    last_message = state['messages'][-1]
    return 'tools' if getattr(last_message, 'tool_calls', None) else 'end'

tool_node = ToolNode([list_unread_emails, summarize_email])

builder = StateGraph(ChatState)
builder.add_node('llm', llm_node)
builder.add_node('tools', tool_node)
builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

graph = builder.compile()

if __name__ == '__main__':
    state = {'messages': []}

    print('Type an instructuon or "quit". \n')

    while True:
        user_message = input('> ')

        if user_message.lower() == 'quit':
            break

        state["messages"].append({'role': 'user', 'content': user_message})
        state = graph.invoke(state)
        print(state['messages'][-1].content)
