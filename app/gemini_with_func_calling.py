import google.generativeai as genai
import os
import json
import datetime
import asyncio
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
    FunctionCall, # Import FunctionCall
)
from typing import List, Dict

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

begin_sentence = "Welcome to our restaurant! I can help you place an order. What would you like to order today?"
ending_sentence = "Thank you for choosing our restaurant. Have a great day!"

agent_prompt = """
As a friendly restaurant order assistant, your role is to help customers place their orders efficiently and accurately.
You should be knowledgeable about the menu, able to answer questions about dishes, and help customers make their selections.
You should also be able to handle special requests and dietary restrictions when possible.

Your main tasks are:
1. Greet customers warmly and help them navigate the menu
2. Take orders accurately, including any special requests or modifications
3. Confirm order details with customers
4. Process payments and provide order confirmation
5. Handle any questions about menu items, ingredients, or preparation methods
6. Maintain a friendly and professional demeanor throughout the interaction

Conversational Style:
- Be friendly and welcoming while maintaining professionalism
- Use clear, concise language to describe menu items
- Be patient and helpful when customers need more information
- Confirm order details clearly before finalizing
- Handle special requests politely and professionally

Today's date is {}.

##Objective
You are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible

## Style Guardrails
- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don't pack everything you want to say into one utterance.
- [Do not repeat] Don't repeat what's in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.
- [Be conversational] Speak like a human as though you're speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.
- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don't be a pushover.
- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.

## Response Guideline
- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say, then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn't catch that", "some noise", "pardon", "you're coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don't repeat yourself.
- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don't repeat yourself in doing this. You should still be creative, human-like, and lively.
- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.
""".format(datetime.date.today().strftime('%A, %B %d, %Y'))

class LlmClient:
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        genai.configure(api_key=api_key)
        
        self.tools = self.prepare_tools()

        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
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

    def convert_transcript_to_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "model", "parts": [utterance.content]})
            else:
                messages.append({"role": "user", "parts": [utterance.content]})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        messages = [{"role": "user", "parts": [agent_prompt]}] 
        
        transcript_messages = self.convert_transcript_to_messages(request.transcript)
        messages.extend(transcript_messages)

        if request.interaction_type == "reminder_required":
            messages.append({"role": "user", "parts": ["(Now the user has not responded in a while, you would say:)"]})
        
        return messages

    def prepare_tools(self):
        tools = [
            {
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
            },
            {
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
                        "
                        }
                    },
                    "required": ["message", "item_id", "quantity"]
                }
            },
            {
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
            },
            {
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
            },
            {
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
        ]
        return tools

    def _show_menu(self, message: str, category: str = None) -> str:
        menu_text = f"{message}\n"
        if category and category in MENU:
            menu_text += f"\n--- {category.replace('_', ' ').title()} ---\n"
            for item_id, item_details in MENU[category].items():
                menu_text