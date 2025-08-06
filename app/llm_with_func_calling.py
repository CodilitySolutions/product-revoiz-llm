from openai import AsyncOpenAI
import os
import json
import datetime
import asyncio
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List, Dict
import httpx
import re

language_codes = {
    "en-US": "English (United States)",
    "en-IN": "English (India)",
    "en-GB": "English (United Kingdom)",
    "de-DE": "German (Germany)",
    "es-ES": "Spanish (Spain)",
    "es-419": "Spanish (Latin America)",
    "hi-IN": "Hindi (India)",
    "ja-JP": "Japanese (Japan)",
    "pt-PT": "Portuguese (Portugal)",
    "pt-BR": "Portuguese (Brazil)",
    "fr-FR": "French (France)",
    'zh-CN': 'China (Chinese)',
    'ru-RU': 'Russia (Russian)',
    'it-IT': 'Italy (Italian)',
    'ko-KR': 'Korea (Korean)',
    'nl-NL': 'Netherlands (Dutch)',
    'pl-PL': 'Poland (Polish)',
    'tr-TR': 'Turkey (Turkish)',
    'vi-VN': 'Vietnam (Vietnamese)'
}

language_code = os.environ["ORDER_LANG"]

# print(language_codes)
agent_language = language_codes.get(language_code, "Language not supported.")
print('Agent Language: ', agent_language)
# Sample menu data structure
# MENU = {
#     "appetizers": {
#         "spring_rolls": {"name": "Spring Rolls", "price": 5.99, "description": "Crispy vegetable spring rolls with sweet chili sauce"},
#         "wings": {"name": "Chicken Wings", "price": 8.99, "description": "8 pieces of crispy wings with choice of sauce"}
#     },
#     "main_courses": {
#         "pad_thai": {"name": "Pad Thai", "price": 12.99, "description": "Stir-fried rice noodles with tofu, peanuts, and tamarind sauce"},
#         "curry": {"name": "Green Curry", "price": 13.99, "description": "Coconut milk curry with vegetables and choice of protein"}
#     },
#     "desserts": {
#         "ice_cream": {"name": "Ice Cream", "price": 4.99, "description": "Vanilla ice cream with chocolate sauce"},
#         "cheesecake": {"name": "Cheesecake", "price": 6.99, "description": "New York style cheesecake with berry compote"}
#     }
# }
MENU = json.loads(os.environ["MENU_LISTING"])
print('Menu: ', MENU)

# begin_sentence = "Welcome to our restaurant! I can help you place an order. What would you like to order today?"
# ending_sentence = "Thank you for choosing our restaurant. Have a great day!"
begin_sentence = os.environ["BEGIN_SENTENCE"]
print('Begin Sentence: ', begin_sentence)
ending_sentence = os.environ["ENDING_SENTENCE"]
print('Ending Sentence: ', ending_sentence)
order_instructions = os.environ["ORDER_INSTRUCTIONS"]
print('Order Instructions: ', order_instructions)
agent_prompt = f"""
You are a friendly restaurant order assistant. Your job is to help customers place their orders efficiently and accurately.

You should:
- Be knowledgeable about the menu and able to answer questions about menu items
- Help customers make their selections
- Handle special requests and dietary restrictions whenever possible

**Important Context**
- **Order Instructions**: {order_instructions}
- **Menu**: {json.dumps(MENU)}
- **Language**: Conduct the entire session in **{agent_language}** — both your questions and the customer's responses should be in this language. Do not use any other language.
- **Payment Methods**: Accepted payment types are Cash, Credit/Debit Card, and Online transfer.
- **Thank You and Conclusion**:
  - Conclude the interaction by thanking the customer for their time and insights.
  - Clearly state this exact sentence as your final response: **{ending_sentence}**
- **Saving the Order**: Before saying the final sentence, always call the function `save_order()` to save the order.

**Your Tasks**
1. Greet the customer warmly and help them navigate the menu.
2. Take the order accurately, including any special requests or dietary modifications.
3. **When confirming the order, ask for the customer's name and preferred payment method**.
4. Confirm the full order with the customer before ending the conversation.
5. Answer questions about menu items, ingredients, or preparation methods.
6. Maintain a friendly, helpful, and professional tone throughout.
7. Make sure to **repeat the order at least once** and ask for confirmation.
8. Always save the order before concluding the conversation.

**Conversational Style**
- Be friendly and welcoming, but professional.
- Use clear and concise language when describing menu items.
- Be patient and helpful when customers need assistance or more information.
- Confirm all details politely and clearly before finalizing the order.
- Handle special requests with care and a positive attitude.
- Never pronounce or spell out asterisks (*) in the customer’s name.

Today's date is: **{datetime.date.today().strftime('%A, %B %d, %Y')}**
"""

print('agent_prompt: ', agent_prompt)

backend_api_url = os.getenv("BACKEND_API_URL")
print('backend_api_url: ', backend_api_url)

def strip_markdown(text):
    # Remove bold and italics
    return re.sub(r'(\*\*|\*|__|_)', '', text)

class LlmClient:
    call_id: str
    current_order: List
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION_ID"),  # Optional
        )
        self.current_order = []  # Track the current order
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    async def _make_api_call_with_retry(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    print(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    raise
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response
    
    def setCallId(self, call_id: str):
        print('set call_id: ', call_id)
        self.call_id = call_id

    async def saveOrder(self, backend_api_url, order_details):
        if backend_api_url and self.call_id:
            post_url = f"{backend_api_url}/api/get-order-item/{self.call_id}/"
            try:
                async with httpx.AsyncClient() as client:
                    post_response = await client.post(post_url, json=order_details)
                    print("Order POST response:", post_response.status_code, post_response.text)
            except Exception as post_exc:
                print("Failed to POST order:", post_exc)
        else:
            print("No BACKEND_API_URL or call_id found, cannot POST order.")

    def getBackendAPIUrl(self):
        return backend_api_url

    def getCurrentOrder(self):
        order_details = {
            "customer_name": "Anonymous",
            "delivery_address": "",
            "payment_method": "",
            "items": self.current_order,
            "total": sum(item["price"] * item["quantity"] for item in self.current_order),
            "order_time": datetime.datetime.now().isoformat()
        }

        return order_details

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        prompt = [
            {
                "role": "system",
                "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don’t repeat what’s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n'
                + agent_prompt,
            }
        ]
        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    def prepare_functions(self):
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "show_menu",
                    "description": "Show the menu to the customer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to introduce the menu"
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional category to show (appetizers, main_courses, desserts). If not provided, show all categories."
                            }
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_to_order",
                    "description": "Add an item to the customer's order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Confirmation message to the customer about the item being added"
                            },
                            "item_id": {
                                "type": "string",
                                "description": "The ID of the menu item being ordered"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "The quantity of the item being ordered"
                            },
                            "special_instructions": {
                                "type": "string",
                                "description": "Any special instructions or modifications for the item"
                            }
                        },
                        "required": ["message", "item_id", "quantity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_order_summary",
                    "description": "Show the current order summary to the customer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to introduce the order summary"
                            }
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_order",
                    "description": "Save the completed order to the database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Confirmation message about the order being saved"
                            },
                            "customer_name": {
                                "type": "string",
                                "description": "Customer's name for the order"
                            },
                            "delivery_address": {
                                "type": "string",
                                "description": "Delivery address if applicable"
                            },
                            "payment_method": {
                                "type": "string",
                                "description": "Payment method for the order"
                            }
                        },
                        "required": ["message", "customer_name", "payment_method"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_order",
                    "description": "Cancel the current order and clear all items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Confirmation message about the order being cancelled"
                            }
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "End the call only when user explicitly requests it or when the order is complete.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message you will say before ending the call with the customer."
                            }
                        },
                        "required": ["message"]
                    }
                }
            }
        ]
        return functions

    async def draft_response(self, request: ResponseRequiredRequest):
        try:
            prompt = self.prepare_prompt(request)
            func_call = {}
            func_arguments = ""
            
            # Use retry logic for the API call
            stream = await self._make_api_call_with_retry(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=prompt,
                stream=True,
                tools=self.prepare_functions(),
            )

            async for chunk in stream:
                print('chunk======================', chunk)
                if len(chunk.choices) == 0:
                    continue

                if chunk.choices[0].delta.tool_calls:
                    tool_calls = chunk.choices[0].delta.tool_calls[0]
                    if tool_calls.id:
                        if func_call:
                            break
                        func_call = {
                            "id": tool_calls.id,
                            "func_name": tool_calls.function.name or "",
                            "arguments": {},
                        }
                    else:
                        func_arguments += tool_calls.function.arguments or ""

                if chunk.choices[0].delta.content:
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=chunk.choices[0].delta.content,
                        content_complete=False,
                        end_call=False,
                    )
                    response.content = strip_markdown(response.content)
                    yield response

            if func_call:
                func_call["arguments"] = json.loads(func_arguments)
                
                if func_call["func_name"] == "show_menu":
                    print('func_name=show_menu')
                    try:
                        category = func_call["arguments"].get("category")
                        menu_text = "Here's our menu:\n\n"
                        
                        if category and category in MENU:
                            menu_text += f"{category.title()}:\n"
                            for item_id, item in MENU[category].items():
                                menu_text += f"- {item['name']}: ${item['price']:.2f}\n"
                                menu_text += f"  {item['description']}\n"
                        else:
                            for category_name, items in MENU.items():
                                menu_text += f"{category_name.title()}:\n"
                                for item_id, item in items.items():
                                    menu_text += f"- {item['name']}: ${item['price']:.2f}\n"
                                    menu_text += f"  {item['description']}\n"
                                menu_text += "\n"

                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=func_call["arguments"]["message"],
                            content_complete=False,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=menu_text,
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                    except Exception as e:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error showing menu: {str(e)}",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                elif func_call["func_name"] == "add_to_order":
                    print('func_name=add_to_order')
                    try:
                        print("Add to order arguments:", func_call["arguments"])
                        item_id = func_call["arguments"]["item_id"]
                        quantity = func_call["arguments"]["quantity"]
                        special_instructions = func_call["arguments"].get("special_instructions", "")

                        item = None
                        for category in MENU.values():
                            if item_id in category:
                                item = category[item_id]
                                break
                        
                        if item:
                            found = False
                            for order_item in self.current_order:
                                if order_item["item_id"] == item_id and order_item.get("special_instructions", "") == special_instructions:
                                    order_item["quantity"] += quantity
                                    found = True
                                    break
                            if not found:
                                order_item = {
                                    "item_id": item_id,
                                    "name": item["name"],
                                    "price": item["price"],
                                    "quantity": quantity,
                                    "special_instructions": special_instructions
                                }
                                self.current_order.append(order_item)
                            
                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=func_call["arguments"]["message"],
                                content_complete=False,
                                end_call=False,
                            )
                            response.content = strip_markdown(response.content)
                            yield response

                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=f"Added {quantity}x {item['name']} to your order.",
                                content_complete=True,
                                end_call=False,
                            )
                            response.content = strip_markdown(response.content)
                            yield response
                        else:
                            raise ValueError(f"Item {item_id} not found in menu")

                    except Exception as e:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error adding item to order: {str(e)}",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                elif func_call["func_name"] == "show_order_summary":
                    print('func_name=show_order_summary')
                    try:
                        if not self.current_order:
                            summary = "Your order is currently empty."
                        else:
                            total = sum(item["price"] * item["quantity"] for item in self.current_order)
                            summary = "Here's your current order:\n"
                            for item in self.current_order:
                                summary += f"- {item['quantity']}x {item['name']} (${item['price']:.2f} each)\n"
                                if item["special_instructions"]:
                                    summary += f"  Special instructions: {item['special_instructions']}\n"
                            summary += f"\nTotal: ${total:.2f}"

                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=func_call["arguments"]["message"],
                            content_complete=False,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=summary,
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                    except Exception as e:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error showing order summary: {str(e)}",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                elif func_call["func_name"] == "save_order":
                    print('func_name=save_order')
                    try:
                        order_details = {
                            "customer_name": func_call["arguments"]["customer_name"],
                            "delivery_address": func_call["arguments"].get("delivery_address", ""),
                            "payment_method": func_call["arguments"]["payment_method"],
                            "items": self.current_order,
                            "total": sum(item["price"] * item["quantity"] for item in self.current_order),
                            "order_time": datetime.datetime.now().isoformat()
                        }
                        print("Saving order:", json.dumps(order_details, indent=2))

                        # Post order to backend
                        await self.saveOrder(backend_api_url, order_details)
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=func_call["arguments"]["message"],
                            content_complete=False,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                        response = ResponseResponse(
                            response_id=request.response_id,
                            content="Order saved successfully! Thank you for your order.",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response
                    except Exception as e:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error saving order: {str(e)}",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                elif func_call["func_name"] == "cancel_order":
                    print('func_name=cancel_order')
                    try:
                        self.current_order = []
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=func_call["arguments"]["message"],
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response
                    except Exception as e:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error cancelling order: {str(e)}",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                elif func_call["func_name"] == "end_call":
                    print('func_name=end_call')
                    try:
                        order_details = {
                            "customer_name": func_call["arguments"]["customer_name"],
                            "delivery_address": func_call["arguments"].get("delivery_address", ""),
                            "payment_method": func_call["arguments"]["payment_method"],
                            "items": self.current_order,
                            "total": sum(item["price"] * item["quantity"] for item in self.current_order),
                            "order_time": datetime.datetime.now().isoformat()
                        }
                        print("Saving order:", json.dumps(order_details, indent=2))

                        # Post order to backend
                        await self.saveOrder(backend_api_url, order_details)
                    except Exception as e:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error saving order: {str(e)}",
                            content_complete=True,
                            end_call=False,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                    ending_message = func_call["arguments"]["message"]
                    # First send the ending message
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=ending_message,
                        content_complete=True,
                        end_call=False,
                    )
                    response.content = strip_markdown(response.content)
                    yield response
                    
                    # Then send a final goodbye message and end the call
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=ending_sentence,
                        content_complete=True,
                        end_call=True,
                    )
                    response.content = strip_markdown(response.content)
                    yield response
            else:
                print('NO FUNC_CALL DETETCTED =================', response.content)
                response = ResponseResponse(
                    response_id=request.response_id,
                    content="",
                    content_complete=True,
                    end_call=False,
                )
                response.content = strip_markdown(response.content)
                yield response
                
        except Exception as e:
            print(f"Error in draft_response: {str(e)}")
            response = ResponseResponse(
                response_id=request.response_id,
                content="I apologize, but I encountered an error. Please try again.",
                content_complete=True,
                end_call=False,
            )
            response.content = strip_markdown(response.content)
            yield response
