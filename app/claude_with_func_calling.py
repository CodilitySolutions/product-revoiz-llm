import os
import random
import datetime
import json
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from anthropic import AsyncAnthropic
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

################################PROMPT########################################

begin_sentence = "Welcome to our restaurant! I can help you place an order. What would you like to order today?"

# Sample menu data structure
MENU = {
    "appetizers": {
        "spring_rolls": {"name": "Spring Rolls", "price": 5.99, "description": "Crispy vegetable spring rolls with sweet chili sauce"},
        "wings": {"name": "Chicken Wings", "price": 8.99, "description": "8 pieces of crispy wings with choice of sauce"}
    },
    "main_courses": {
        "pad_thai": {"name": "Pad Thai", "price": 12.99, "description": "Stir-fried rice noodles with tofu, peanuts, and tamarind sauce"},
        "curry": {"name": "Green Curry", "price": 13.99, "description": "Coconut milk curry with vegetables and choice of protein"}
    },
    "desserts": {
        "ice_cream": {"name": "Ice Cream", "price": 4.99, "description": "Vanilla ice cream with chocolate sauce"},
        "cheesecake": {"name": "Cheesecake", "price": 6.99, "description": "New York style cheesecake with berry compote"}
    }
}

role = """
As a friendly restaurant order assistant, your role is to help customers place their orders efficiently and accurately.
You should be knowledgeable about the menu, able to answer questions about dishes, and help customers make their selections.
You should also be able to handle special requests and dietary restrictions when possible.

Today's date is {}.
""".format(datetime.date.today().strftime('%A, %B %d, %Y'))

task = """
Your main tasks are:
1. Greet customers warmly and help them navigate the menu
2. Take orders accurately, including any special requests or modifications
3. Confirm order details with customers
4. Process payments and provide order confirmation
5. Handle any questions about menu items, ingredients, or preparation methods
6. Maintain a friendly and professional demeanor throughout the interaction
"""

conversational_style = """
- Be friendly and welcoming while maintaining professionalism
- Use clear, concise language to describe menu items
- Be patient and helpful when customers need more information
- Confirm order details clearly before finalizing
- Handle special requests politely and professionally
"""

agent_prompt = """
<agent_prompt>

<role>
{}
</role>

<task>
{}
</task>

<conversational_style>
{}
</conversational_style>

</agent_prompt>
""".format(role, task, conversational_style)


style_guardrails = """
- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don't pack everything you want to say into one utterance.
- [Do not repeat] Don't repeat what's in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.
- [Be conversational] Speak like a human as though you're speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.
- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don't be a pushover.
- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.
"""

response_guideline = """
- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. 
When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn't catch that", "some noise", "pardon", "you're coming through choppy", "static in your speech", "voice is cutting in and out"). 
Do not ever mention "transcription error", and don't repeat yourself.
- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don't repeat yourself in doing this. You should still be creative, human-like, and lively.
- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.
"""

additional_scenarios = """

"""

system_prompt = """

<system_prompt>

<style_guardrails>
{}
</style_guardrails>

<response_guideline>
{}
</response_guideline>

<agent_prompt>
{}
</agent_prompt>

<scenarios_handling>
{}
</scenarios_handling>

</system_prompt>
""".format(style_guardrails, response_guideline, agent_prompt, additional_scenarios)


########################################################################
class LlmClient:
    def __init__(self):
        # self.client = AsyncOpenAI(
        #     api_key=os.environ["OPENAI_API_KEY"],
        # )
        self.client = AsyncAnthropic() 

    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response


    def convert_transcript_to_anthropic_messages(self, transcript: List[Utterance]):
        messages = [
            {"role": "user", "content": 
             """
             ...
             """},

        ]
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                if utterance.content.strip():
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] += " " + utterance.content
                    else:
                        messages.append({"role": "user", "content": utterance.content})
                else:
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] += " ..."
                    else:
                        messages.append({"role": "user", "content": "..."})

        return messages


    def prepare_prompt(self, request: ResponseRequiredRequest, func_result=None):
        prompt = []
        # print(f"Request transcript: {request.transcript}")
        transcript_messages = self.convert_transcript_to_anthropic_messages(
            request.transcript
        )
        # print(f"Transcript messages: {transcript_messages}")

        for message in transcript_messages:
            prompt.append(message)

        if func_result:
            # add function call to prompt
            prompt.append({
                "role": "assistant",
                "content": [
                    {
                        "id": func_result["id"],
                        "input": func_result["arguments"],
                        "name": func_result["func_name"],
                        "type": "tool_use"
                    }
                ]
            })

            # add function call result to prompt
            tool_result_content = {
                "type": "tool_result",
                "tool_use_id": func_result["id"],
                "content": func_result["result"] or ''
            }
            
            if "is_error" in func_result:
                tool_result_content["is_error"] = func_result["is_error"]
            
            prompt.append({
                "role": "user",
                "content": [tool_result_content]
            })

        # if request.interaction_type == "reminder_required":
        #     prompt.append(
        #         {
        #             "role": "user",
        #             "content": "(Now the user has not responded in a while, you would say:)",
        #         }
        #     )

        # print(f"Prompt: {prompt}")
        return prompt

    # Step 1: Prepare the function calling definition to the prompt
    def prepare_functions(self):
        functions = [
            {
                "name": "show_menu",
                "description": "Show the menu to the customer",
                "input_schema": {
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
            },
            {
                "name": "end_call",
                "description": """
                End the call only when user explicitly requests it or when the order is complete.
                """,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message you will say before ending the call with the customer."
                        },
                        "reason": {
                            "type": "string",
                            "description": "An internal note explaining why the call is being ended at this point."
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "add_to_order",
                "description": "Add an item to the customer's order",
                "input_schema": {
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
            },
            {
                "name": "show_order_summary",
                "description": "Show the current order summary to the customer",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to introduce the order summary"
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "save_order",
                "description": "Save the completed order to the database",
                "input_schema": {
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
        ]
        return functions

    async def draft_response(self, request, func_result=None):
        prompt = self.prepare_prompt(request, func_result)
        print(f"request.response_id: {request.response_id}")

        func_call = {}
        func_arguments = ""
        last_func_name = None
        last_func_args = None
        current_order = []  # Track the current order

        stream = await self.client.messages.create(
            max_tokens=256,
            messages=prompt,
            model="claude-3-haiku-20240307",
            stream=True,
            temperature=0.0,
            tools=self.prepare_functions(),
            tool_choice={"type": "auto"},
            system=system_prompt,
        )

        async for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                content_block = event.content_block
                if content_block.type == "tool_use":
                    tool_use = content_block
                    if tool_use.id:
                        if func_call:
                            break
                        func_call = {
                            "id": tool_use.id,
                            "func_name": tool_use.name or "",
                            "arguments": {},
                        }
                    else:
                        func_arguments = ""

            elif event_type == "content_block_delta":
                delta_type = event.delta.type
                if delta_type == "text_delta":
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=event.delta.text,
                        content_complete=False,
                        end_call=False,
                    )
                    yield response
                elif delta_type == "input_json_delta":
                    func_arguments += event.delta.partial_json or ""

            elif event_type == "message_delta":
                stop_reason = event.delta.stop_reason
                print(f"Stop reason: {stop_reason}")
                if stop_reason == "tool_use":
                    if func_call:
                        func_call["arguments"] = json.loads(func_arguments)
                        if func_call["func_name"] == last_func_name and func_call["arguments"] == last_func_args:
                            continue
                        last_func_name = func_call["func_name"]
                        last_func_args = func_call["arguments"]

                        if func_call["func_name"] == "show_menu":
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
                                yield response

                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": menu_text
                                }

                            except Exception as e:
                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": f"Error: {str(e)}",
                                    "is_error": True
                                }

                        elif func_call["func_name"] == "end_call":
                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=func_call["arguments"]["message"],
                                content_complete=True,
                                end_call=True,
                            )
                            yield response

                        elif func_call["func_name"] == "add_to_order":
                            try:
                                item_id = func_call["arguments"]["item_id"]
                                quantity = func_call["arguments"]["quantity"]
                                
                                # Find the item in the menu
                                item = None
                                for category in MENU.values():
                                    if item_id in category:
                                        item = category[item_id]
                                        break
                                
                                if item:
                                    order_item = {
                                        "item_id": item_id,
                                        "name": item["name"],
                                        "price": item["price"],
                                        "quantity": quantity,
                                        "special_instructions": func_call["arguments"].get("special_instructions", "")
                                    }
                                    current_order.append(order_item)
                                    
                                    response = ResponseResponse(
                                        response_id=request.response_id,
                                        content=func_call["arguments"]["message"],
                                        content_complete=False,
                                        end_call=False,
                                    )
                                    yield response
                                    
                                    func_result = {
                                        "id": func_call["id"],
                                        "arguments": func_call["arguments"],
                                        "func_name": func_call["func_name"],
                                        "result": f"Added {quantity}x {item['name']} to the order."
                                    }
                                else:
                                    raise ValueError(f"Item {item_id} not found in menu")

                            except Exception as e:
                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": f"Error: {str(e)}",
                                    "is_error": True
                                }

                        elif func_call["func_name"] == "show_order_summary":
                            try:
                                if not current_order:
                                    summary = "Your order is currently empty."
                                else:
                                    total = sum(item["price"] * item["quantity"] for item in current_order)
                                    summary = "Here's your current order:\n"
                                    for item in current_order:
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
                                yield response

                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": summary
                                }

                            except Exception as e:
                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": f"Error: {str(e)}",
                                    "is_error": True
                                }

                        elif func_call["func_name"] == "save_order":
                            try:
                                # Here you would typically save to a database
                                # For now, we'll just print the order details
                                order_details = {
                                    "customer_name": func_call["arguments"]["customer_name"],
                                    "delivery_address": func_call["arguments"].get("delivery_address", ""),
                                    "payment_method": func_call["arguments"]["payment_method"],
                                    "items": current_order,
                                    "total": sum(item["price"] * item["quantity"] for item in current_order),
                                    "order_time": datetime.datetime.now().isoformat()
                                }
                                
                                print("Saving order:", json.dumps(order_details, indent=2))
                                
                                response = ResponseResponse(
                                    response_id=request.response_id,
                                    content=func_call["arguments"]["message"],
                                    content_complete=False,
                                    end_call=False,
                                )
                                yield response

                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": "Order saved successfully!"
                                }

                            except Exception as e:
                                func_result = {
                                    "id": func_call["id"],
                                    "arguments": func_call["arguments"],
                                    "func_name": func_call["func_name"],
                                    "result": f"Error: {str(e)}",
                                    "is_error": True
                                }

                        # Continue drafting the response after function calls
                        async for response in self.draft_response(request, func_result):
                            yield response

            elif event_type == "message_stop":
                response = ResponseResponse(
                    response_id=request.response_id,
                    content="",
                    content_complete=True,
                    end_call=False,
                )
                yield response
