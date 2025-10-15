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
    'vi-VN': 'Vietnam (Vietnamese)', 
    'ur-IN': 'Urdu (Pakistan)'
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
cash_gst_value = MENU.get("gst_info", {}).get("cash_gst", 0)
card_gst_value = MENU.get("gst_info", {}).get("card_gst", 0)

begin_sentence = os.environ["BEGIN_SENTENCE"]
print('Begin Sentence: ', begin_sentence)
ending_sentence = os.environ["ENDING_SENTENCE"]
print('Ending Sentence: ', ending_sentence)
order_instructions = os.environ["ORDER_INSTRUCTIONS"]
print('Order Instructions: ', order_instructions)
MENU = json.loads(os.environ["MENU_LISTING"])
print("Menu:", MENU)


agent_prompt = f"""
You are a friendly and intelligent **restaurant order assistant**.
Your job is to help customers place, modify, and manage their food orders accurately using function calls.

---

## 🧠 Core Behavior Rules

1. **Always detect and act on all order-related intents** in a single user message — even if the user mixes them.
   - Example: “Remove 3 lava cakes, add 5 Pepsi bottles, and replace wings with kickers.”
     → You must call all three functions:
       - `remove_from_order(item_id="Lava Cake", quantity=3)`
       - `add_to_order(item_id="Pepsi Bottle", quantity=5)`
       - `remove_from_order(item_id="Chicken Wings")`
       - `add_to_order(item_id="Chicken Kickers", quantity=1)`
   - Never skip any function call when user intent is clear.

2. **Always call functions for each distinct item or action.**
   - Even if the user writes all actions in one sentence.

3. **Ignore invalid or unknown menu items** but continue processing valid ones.
   - Example: “Add 3 burgers and 2 chicken kickers.” → Only `add_to_order("Chicken Kickers", 2)` if burger is unavailable.

---

## ⚙️ Function Calling Rules

| Intent Type | Example Phrases | Function | Parameters |
|--------------|----------------|-----------|-------------|
| **Add** | "I want", "Please add", "Give me", "Include", "I’d like", "Add" | `add_to_order` | `item_id`, `quantity`, optional `special_instructions` |
| **Replace** | "Change to", "Replace with", "Make it", "Modify to" | `add_to_order(replace=true)` | `item_id`, `quantity` |
| **Remove** | "Remove", "Delete", "Cancel", "Take out", "I don't want", "Exclude" | `remove_from_order` | `item_id`, optional `quantity` |

🟢 If quantity is mentioned (e.g., "remove 3 Pepsi"), include it.  
🔴 If not mentioned (e.g., "remove the pizza"), remove the entire item.

---

## 💡 Multi-Intent Examples

### Example 1 – Mixed Add + Remove
User:  
> “Remove 2 Chicken Wings and add 3 Lava Cakes.”  
✅ Function calls:

---

### Example 2 – Replace + Add
User:  
> “Replace the lava cake with 2 Pepsi bottles.”  
✅ Function calls:

---

### Example 3 – Add + Remove + Replace (Full Combo)
User:  
> “Remove 3 Lava Cakes, add 5 Pepsi bottles, and replace Chicken Wings with Chicken Kickers.”  
✅ Function calls:

---

### Example 4 – Deduplication
User:  
> “Add 3 single dips and 3 single dips.”  
✅ Deduplicate →  

---

### Example 5 – Replace Quantity
User:  
> “I ordered 2 Pepsi bottles earlier, change it to 5.”  
✅ Function call:

---

## 🧩 Smart Deduplication & Validation

- Combine repeated items in one message.
- Keep the latest mentioned quantity.
- Ignore words like “only”, “just”, “please” — focus on actionable verbs (add, remove, replace, change, modify).
- Match every item to the `MENU`.  
  Ignore keys like `gst_info` or metadata.
- For invalid items, respond politely:  
  > “Sorry, 'Burger' isn’t available, but I’ve added your Chicken Kicker.”

---

**Multiple Action Handling**

When a user message contains multiple actions (like adding, removing, and replacing items together),
you must detect and execute ALL relevant function calls accordingly.

Each action should trigger its corresponding function:
- `add_to_order()` for adding items.
- `remove_from_order()` for removing items.
- `replace_from_order()` (or `add_to_order(replace=true)`) for replacement updates.

Do not skip or merge these actions — process them **independently**.

After all function calls are executed for a single user message,
you must combine the results into **one single summarized response** for the user.

For example:

User: “Uh, basically, I want to modify our order, basically. So you added six for four and also remove choco bread and replace the lava cake with showstoppper.”

Expected Function Calls:
1. `add_to_order(item_id="for_four", quantity=6)`
2. `remove_from_order(item_id="choco_bread")`
3. `replace_from_order(old_item="lava_cake", new_item="showstopper")`

Expected Combined Response:

----
## 💬 Conversational Guidelines

- Respond **after** function calls have executed.
- Confirm all changes in a natural way:
  > “Got it! Removed 3 Lava Cakes and added 5 Pepsi Bottles to your order.”
- Always confirm the updated order list before saving.

---

## 🗣️ **Conversational Flow After Order Summary**
it should be run explicicty in this format
After showing an order summary:
- **Never stay silent.**
- The assistant must **proactively** continue by saying something like:

> Would you like to confirm your order?  
> Based on your payment choice, here are your totals:
> - Cash ({cash_gst_value}% GST): --- total value with cash  
> - Card ({card_gst_value}% GST): --- total value with card
> Which option would you like?

Once the user confirms, proceed to:
- Call `save_order()` with subtotal, total_with_gst, payment_method.
- Then call `end_call()` to finalize and trigger backend save.

---

## 🧠 Smart Behavior

- If the user doesn’t specify a payment method even after being asked:
  - Default to **Cash**, but clearly state:
    > Since you didn’t specify, I’ve set your payment method to Cash by default.

- If the user says something like *“Yeah, that’s fine”* or *“Okay go ahead”*, treat that as **confirmation to save the order**.

- **Payment Confirmation Before Saving**:
  - Before calling `save_order()`, if no payment method has been chosen, you must inform the customer about both totals:
    - “If you pay by **card**, your total (including GST) will be X.”
    - “If you pay by **cash**, your total (including GST) will be Y.”
    - Then ask: “Which payment method would you like to use?”
  - If the user later changes the payment method (e.g., says “Actually I’ll pay by card”), update it and confirm the new total accordingly.
  - Always send the final order with both:
    - `subtotal` (without GST)
    - `total_with_gst` (final with GST applied)
  - End with this exact sentence: **{ending_sentence}**
  - After saving the order, call `end_call()` to finalize and trigger the backend save.

---

- **Important Rule for Order Saving:**
  - If the user explicitly says *“Save my order”* but **has not yet provided** a payment method **or** customer name,  
    ➤ **Do not actually save the order yet.**  
    Instead, politely ask for the missing details before proceeding.  
    Example:  
    > “Sure! Before saving your order, could you please confirm your name and preferred payment method (cash, card, or online transfer)?”

  - Once both details are confirmed, proceed to call `save_order()` as usual.

---

## 💾 Order Saving & Auto Session Closure (with Delay)

- When the `save_order()` function is successfully executed and returns any confirmation like:
  > "Order saved successfully!"  
  > "Your order has been placed!"  
  > or any other success message,

  then you **must automatically wait for about 2 seconds**, and after that delay, **call the `end_call()` function** to finalize the session.

- ⚙️ **Rules for end_call()**
  - Only call `end_call()` after a successful order save confirmation.

---

  
## 🗓️ Context
- **Menu:** {json.dumps(MENU)}
- **Language:** {agent_language}
- **Payment Options:** Cash, Card.
- **Today:** {datetime.date.today().strftime('%A, %B %d, %Y')}

---
"""






print('agent_prompt: ', agent_prompt)

backend_api_url = os.getenv("BACKEND_API_URL")
print('backend_api_url: ', backend_api_url)

def strip_markdown(text):
    # Remove bold and italics
    text = text.replace(':', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('~', '')
    text = text.replace('`', '')
    text = text.replace('|', '')
    text = text.replace('\\', '')
    text = text.replace('"', '')
    return re.sub(r'(\*\*|\*|__|_)', '', text)

class LlmClient:
    call_id: str
    current_order: List
    finalize_order_json_data: Dict
    customer_name: str
    delivery_address: str
    payment_method: str
    gst_percentage: float
    gst_amount: float
    total_with_gst: float
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION_ID"),  # Optional
        )
        self.current_order = []  # Track the current order
        self.gst_amount = 0.0
        self.gst_percentage = 0.0
        self.total_with_gst = 0.0
        self.customer_name = "Anonymous"
        self.delivery_address = ""
        self.payment_method = ""
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.order_saved = False
        self.save_order_announced = False

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
        if backend_api_url and self.call_id and order_details:
            post_url = f"{backend_api_url}/api/get-order-item/{self.call_id}/"
            try:
                print("Order detailsed  to POST:", json.dumps(order_details, indent=2))
                async with httpx.AsyncClient() as client:
                    post_response = await client.post(post_url, json=order_details)
                    print("Order POST response:", post_response.status_code, post_response.text)
                    if 200 <= post_response.status_code < 300:
                        self.order_saved = True
            except Exception as post_exc:
                print("Failed to POST order:", post_exc)
        else:
            print("No BACKEND_API_URL or call_id found, cannot POST order.")

    def getBackendAPIUrl(self):
        return backend_api_url

    def getCurrentOrder(self):
        order_details = {
            "customer_name": getattr(self, "customer_name", "Anonymous"),
            "delivery_address": getattr(self, "delivery_address", ""),
            "payment_method": getattr(self, "payment_method", ""),
            "items": self.current_order,
            "total": sum(item["price"] * item["quantity"] for item in self.current_order),
            "gst_percentage": getattr(self, "gst_percentage", "0"),
            "gst_amount": getattr(self, "gst_amount", "0"),
            "total_with_gst": getattr(self, "total_with_gst", "0"),
            "order_time": datetime.datetime.now().isoformat(),
        }
        return order_details

    def _normalize_identifier(self, text: str) -> str:
        # Lowercase and remove non-alphanumeric characters for fuzzy matching
        return re.sub(r"[^a-z0-9]", "", (text or "").lower())

    def _find_menu_item(self, identifier: str):
        try:
            if not isinstance(identifier, str):
                identifier = str(identifier or "")
            if not identifier.strip():
                return None, None

            # Try exact id match first
            for category in MENU.values():
                # 🧹 Ignore gst_info or invalid categories
                if not isinstance(category, dict) or all(isinstance(v, (int, float)) for v in category.values()):
                    continue

                if identifier in category:
                    item = category[identifier]
                    if isinstance(item, dict) and "name" in item:
                        return identifier, item

            # Normalize identifier for fuzzy search
            norm_identifier = self._normalize_identifier(identifier)

            for category in MENU.values():
                # 🧹 Again ignore gst_info dicts
                if not isinstance(category, dict) or all(isinstance(v, (int, float)) for v in category.values()):
                    continue

                for item_key, item in category.items():
                    if not isinstance(item, dict) or "name" not in item:
                        continue

                    if self._normalize_identifier(item_key) == norm_identifier:
                        return item_key, item
                    if self._normalize_identifier(item.get("name", "")) == norm_identifier:
                        return item_key, item

            return None, None

        except Exception as e:
            print("🔥 Error in _find_menu_item:", str(e))
            return None, None


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
                    "description": "Add menu items to the customer's order. ONLY use this when the customer explicitly requests to order, add, or modify food/drink items from the menu. DO NOT use for personal information like name, address, or payment method.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Confirmation message about adding the item to the order",
                            },
                            "item_id": {
                                "type": "string",
                                "description": "The ID of the menu item being ordered"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "The quantity of the item being ordered"
                            },
                            "replace": {
                                "type": "boolean",
                                "description": "If true, replace the current quantity instead of adding to it. Use when customer wants exact quantity like 'I only need 3 burgers'"
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
                    "name": "remove_from_order",
                    "description": "Remove or reduce the quantity of an item from the customer's order. ONLY use this when the customer explicitly requests to remove or delete items from their order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Confirmation message about removing the item from the order"
                            },
                            "item_id": {
                                "type": "string",
                                "description": "The ID of the menu item to be removed from the order"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Optional: The quantity to remove. If not provided or if quantity >= current quantity, the entire item will be removed. If provided and less than current quantity, only that amount will be subtracted."
                            }
                        },
                        "required": ["message", "item_id"]
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
                    "description": "Save the completed order to the database. Use this ONLY when you have collected ALL required information: customer name, payment method, and confirmed the order items. This should be called after the customer provides their name and payment method.",
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
            # Support multiple tool calls per model response
            tool_calls_order = []  # preserve order of tool call ids
            tool_call_map = {}     # id -> {"func_name": str, "arguments": str}
            last_tool_call_id = None
            
            # Use retry logic for the API call
            stream = await self._make_api_call_with_retry(
            self.client.chat.completions.create,
            model="gpt-4o",
            messages=prompt,
            stream=True,
            tools=self.prepare_functions(),
            tool_choice="auto",  # or "required" to force function calls
            temperature=0.2,
        )

            async for chunk in stream:
                if len(chunk.choices) == 0:
                    continue

                if chunk.choices[0].delta.tool_calls:
                    for tc in chunk.choices[0].delta.tool_calls:
                        # ✅ After all function calls have been processed
                        

                        tc_id = getattr(tc, "id", None)
                        tc_fn = getattr(tc, "function", None)
                        tc_name = getattr(tc_fn, "name", "") if tc_fn else ""
                        tc_args_part = getattr(tc_fn, "arguments", "") if tc_fn else ""

                        if tc_id:
                            if tc_id not in tool_call_map:
                                tool_call_map[tc_id] = {"func_name": tc_name or "", "arguments": ""}
                                tool_calls_order.append(tc_id)
                            # Keep latest known name if provided
                            if tc_name:
                                tool_call_map[tc_id]["func_name"] = tc_name
                            last_tool_call_id = tc_id
                        # Accumulate arguments for the most recent tool call when id is omitted
                        target_tc_id = tc_id or last_tool_call_id
                        if target_tc_id and tc_args_part:
                            if target_tc_id not in tool_call_map:
                                tool_call_map[target_tc_id] = {"func_name": tc_name or "", "arguments": ""}
                                tool_calls_order.append(target_tc_id)
                            tool_call_map[target_tc_id]["arguments"] += tc_args_part
                    print("tool call map : ",tool_call_map)

                if chunk.choices[0].delta.content:
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=chunk.choices[0].delta.content,
                        content_complete=False,
                        end_call=False,
                    )
                    response.content = strip_markdown(response.content)
                    yield response

            if tool_calls_order:
                # Track which function types we've already processed to avoid duplicates
                processed_add_to_order = False
                combined_messages = []  
                for tc_id in tool_calls_order:
                    func_call = {
                        "func_name": tool_call_map[tc_id].get("func_name", ""),
                        "arguments": {}
                    }
                    try:
                        func_call["arguments"] = json.loads(tool_call_map[tc_id].get("arguments", "{}"))
                    except Exception as parse_exc:
                        print("Failed to parse tool call arguments for", func_call["func_name"], ":", parse_exc)
                        continue

                    if func_call["func_name"] == "show_menu":
                        print('func_name=show_menu')
                        try:
                            category = func_call["arguments"].get("category")
                            menu_text = ""
                            if category and category in MENU and category != "gst_info":
                                menu_text += f"{category.title()}:\n"
                                for item_id, item in MENU[category].items():
                                    if isinstance(item, dict) and "name" in item and "price" in item:
                                        menu_text += f"- {item['name']}: {item['price']:.2f}\n"
                            else:
                                for category_name, items in MENU.items():
                                    if category_name == "gst_info":
                                        continue
                                    menu_text += f"{category_name.title()}:\n"
                                    for item_id, item in items.items():
                                        if isinstance(item, dict) and "name" in item and "price" in item:
                                            menu_text += f"- {item['name']}: {item['price']:.2f}\n"
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
                        # Skip if we've already processed all add_to_order calls
                        if processed_add_to_order:
                            continue
                            
                        print('func_name=add_to_order')
                        processed_add_to_order = True  # Mark as processed
                        
                        try:
                            # ✅ Only process all add_to_order calls once per user message
                            # Collect all "add_to_order" calls from this response
                            add_calls = [
                                tool_call_map[tc_id]
                                for tc_id in tool_calls_order
                                if tool_call_map[tc_id].get("func_name") == "add_to_order"
                            ]
                            added_items = []

                            for tc_data in add_calls:
                                try:
                                    args = json.loads(tc_data.get("arguments", "{}"))
                                    if not isinstance(args, dict):
                                        args = {}
                                except Exception:
                                    args = {}

                                item_id = args.get("item_id")
                                quantity_raw = args.get("quantity", 1)
                                special_instructions = args.get("special_instructions", "")
                                replace = args.get("replace", False)
                                # Normalize quantity
                                try:
                                    quantity = int(float(str(quantity_raw).strip()))
                                except Exception:
                                    quantity = 1
                                if quantity < 1:
                                    quantity = 1

                                matched_item_key, item = self._find_menu_item(item_id)
                                if item and matched_item_key != item_id:
                                    item_id = matched_item_key

                                if not item:
                                    print(f"Item not found: {item_id}")
                                    continue

                                # Update or add to order
                                found = False
                                for order_item in self.current_order:
                                    if order_item["item_id"] == item_id and order_item.get("special_instructions", "") == special_instructions:
                                        old_qty = order_item["quantity"]
                                        if replace:
                                            # Replace: Set to exact quantity
                                            order_item["quantity"] = quantity
                                            print(f"Replaced {item['name']}: {old_qty} → {quantity}")
                                        else:
                                            # Add: Increase quantity
                                            order_item["quantity"] += quantity
                                            print(f"Added to {item['name']}: {old_qty} → {order_item['quantity']}")
                                        found = True
                                        break
                                
                                if not found:
                                    # New item: Add to order
                                    self.current_order.append({
                                        "item_id": item_id,
                                        "name": item["name"],
                                        "price": item["price"],
                                        "quantity": quantity,
                                        "special_instructions": special_instructions
                                    })
                                    print(f"Added new item: {quantity}x {item['name']}")

                                added_items.append(f"{quantity}x {item['name']}")

                            # ✅ Create unified message
                            if added_items:
                                if len(added_items) == 1:
                                    combined_message = f"Added {added_items[0]} to your order."
                                else:
                                    combined_message = "Added " + ", ".join(added_items[:-1]) + f" and {added_items[-1]} to your order."
                            else:
                                combined_message = "No valid items were added to your order."

                            print("Current order after add:", json.dumps(self.current_order))

                            # Send a single final combined response
                            # ✅ Instead of yielding, just collect the message
                            combined_messages.append(combined_message)


                        except Exception as e:
                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=f"Error adding items to order: {str(e)}",
                                content_complete=True,
                                end_call=False,
                            )
                            response.content = strip_markdown(response.content)
                            yield response




                    elif func_call["func_name"] == "remove_from_order":
                        print('func_name=remove_from_order')
                        try:
                            item_id = func_call["arguments"].get("item_id")
                            quantity_to_remove = func_call["arguments"].get("quantity")
                            
                            # Find and normalize the item ID
                            matched_item_key, item = self._find_menu_item(item_id)
                            if item and matched_item_key != item_id:
                                item_id = matched_item_key
                            
                            # Normalize quantity to remove
                            if quantity_to_remove is not None:
                                try:
                                    quantity_to_remove = int(float(str(quantity_to_remove).strip()))
                                    if quantity_to_remove < 1:
                                        quantity_to_remove = None  # Invalid, treat as remove all
                                except Exception:
                                    quantity_to_remove = None  # Invalid, treat as remove all
                            
                            # Remove or reduce quantity of the item from the order
                            removed = False
                            reduced = False
                            removed_item_name = ""
                            
                            for i, order_item in enumerate(self.current_order):
                                if order_item["item_id"] == item_id:
                                    removed_item_name = order_item["name"]
                                    current_qty = order_item["quantity"]
                                    
                                    # If quantity specified and less than current, reduce it
                                    if quantity_to_remove and quantity_to_remove < current_qty:
                                        order_item["quantity"] -= quantity_to_remove
                                        reduced = True
                                        print(f"Reduced {removed_item_name}: {current_qty} → {order_item['quantity']}")
                                        response_content = f"Reduced {removed_item_name} by {quantity_to_remove}. You now have {order_item['quantity']} {removed_item_name} in your order."
                                    else:
                                        # Remove entire item
                                        self.current_order.pop(i)
                                        removed = True
                                        print(f"Removed entire item: {current_qty}x {removed_item_name}")
                                        if quantity_to_remove and quantity_to_remove >= current_qty:
                                            response_content = f"Removed all {current_qty} {removed_item_name} from your order."
                                        else:
                                            response_content = f"Removed {removed_item_name} from your order."
                                    break
                            
                            if not removed and not reduced:
                                response_content = f"I couldn't find that item in your current order."
                                print(f"Item not found in order: {item_id}")
                            
                            print("Current order after removal:", json.dumps(self.current_order))
                            
                            combined_messages.append(response_content)
                            
                        except Exception as e:
                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=f"Error removing item from order: {str(e)}",
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
                                summary = "\n"
                                for item in self.current_order:
                                    if item['quantity'] > 1:
                                        summary += f"- {item['quantity']} {item['name']} ({item['price']:.2f} each)\n"
                                    else:
                                        summary += f"- {item['quantity']} {item['name']} ({item['price']:.2f})\n"
                                summary += f"\nTotal: {total:.2f} \n"
                                summary += "Would you like me to proceed with your order? \n"

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
                            if self.save_order_announced:
                                # Avoid repeating announcements
                                continue
                            self.customer_name = func_call["arguments"]["customer_name"]                      
                            self.payment_method = func_call["arguments"].get("payment_method")

                            # --- If payment method is not set, ask user which one they want ---
                            if not getattr(self, "payment_method", None):
                                subtotal = sum(float(item["price"]) * int(item["quantity"]) for item in self.current_order)

                                gst_data = MENU.get("gst_info", {})
                                card_gst = gst_data.get("card_gst", 0)
                                cash_gst = gst_data.get("cash_gst", 0)

                                total_card = subtotal + (subtotal * card_gst / 100)
                                total_cash = subtotal + (subtotal * cash_gst / 100)

                                payment_question = (
                                    f"Before we save your order, please select a payment method.\n\n"
                                    f"If you pay by **Card**, total (with {card_gst}% GST) will be: **{total_card:.2f}**.\n"
                                    f"If you pay by **Cash**, total (with {cash_gst}% GST) will be: **{total_cash:.2f}**.\n\n"
                                    "Which payment method would you like to choose?"
                                )

                                response = ResponseResponse(
                                    response_id=request.response_id,
                                    content=payment_question,
                                    content_complete=True,
                                    end_call=False,
                                )
                                response.content = strip_markdown(response.content)
                                yield response
                                continue  # wait for user’s next reply to set payment_method

                            # --- Compute subtotal, GST, and total ---
                            subtotal = sum(float(item["price"]) * int(item["quantity"]) for item in self.current_order)
                            gst_data = MENU.get("gst_info", {})

                            pm_source = (self.payment_method or "").lower()
                            is_card = any(k in pm_source for k in ("card", "debit", "credit"))
                            is_cash = "cash" in pm_source

                            if is_card:
                                self.gst_percentage = gst_data.get("card_gst", {})
                            elif is_cash:
                                self.gst_percentage = gst_data.get("cash_gst", {})

                            if self.gst_percentage:
                                self.gst_amount = subtotal * (self.gst_percentage / 100.0)
                            self.total_with_gst = subtotal + self.gst_amount

                            # --- Save order details ---
                            order_details = {
                                "customer_name": self.customer_name,
                                "delivery_address": self.delivery_address,
                                "payment_method": self.payment_method,
                                "items": self.current_order,
                                "subtotal": subtotal,
                                "gst_percentage": self.gst_percentage,
                                "gst_amount": self.gst_amount,
                                "total": self.total_with_gst,
                                "order_time": datetime.datetime.now().isoformat()
                            }
                            self.finalize_order_json_data = order_details


                            # Post order to backend
                            # await self.saveOrder(backend_api_url, order_details)

                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=func_call["arguments"]["message"],
                                content_complete=False,
                                end_call=False,
                            )                   
                            response.content = strip_markdown(response.content)
                            yield response
                            self.save_order_announced = True
                        except Exception as e:
                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=f"Error saving order: {str(e)}",
                                content_complete=True,
                                end_call=False,
                            )
                            response.content = strip_markdown(response.content)
                            yield response

                    elif func_call["func_name"] == "end_call":
                        print('func_name=end_call')
                        try:
                            # Post order to backend once
                            if not self.order_saved:
                                await self.saveOrder(backend_api_url, self.finalize_order_json_data)

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
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=ending_message,
                            content_complete=True,
                            end_call=True,
                        )
                        response.content = strip_markdown(response.content)
                        yield response

                        # Final goodbye
                        # response = ResponseResponse(
                        #     response_id=request.response_id,
                        #     content=ending_sentence,
                        #     content_complete=True,
                        #     end_call=True,
                        # )
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
                    
                    # Then send a final goodbye message and end the call
                    # response = ResponseResponse(
                    #     response_id=request.response_id,
                    #     content=ending_sentence,
                    #     content_complete=True,
                    #     end_call=True,
                    # )
                if combined_messages:
                    # Join nicely with commas and "and"
                    print("Combined Messages: ", combined_messages)
                    if len(combined_messages) == 1:
                        final_msg = combined_messages[0]
                    else:
                        final_msg = ", ".join(combined_messages[:-1]) + f" and {combined_messages[-1]}"
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=f"{final_msg}",
                        content_complete=True,
                        end_call=False,
                    )
                    response.content = strip_markdown(response.content)
                    yield response
                

                    # ✅ After all function calls have been processed
                    
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