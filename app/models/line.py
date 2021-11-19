
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

line_bot_api = LineBotApi('Q3avjULQUNRb4HywTXE8fM8/ygkuCM60j+0Bs1JO976+s96E/R90rAJP+CQnGgiyMfw5GLYPamDDSDBZgwgek137Euq92IC417n7KTwJxy7EQs9AxJY9oLtq2G5WQSCgmekjtVLkT0euoapXAaFCuwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('1b07730e6d130bee9e720d188e81e8df')


def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))