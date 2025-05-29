import os
from openai import OpenAI

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
}

translations = {
    "en-US": "Hey there, I'm your AI interviewer Robin. Thank you for taking the time for this interview. Are you ready to begin?",
    "en-IN": "Hey there, I'm your AI interviewer Robin. Thank you for taking the time for this interview. Are you ready to begin?",
    "en-GB": "Hey there, I'm your AI interviewer Robin. Thank you for taking the time for this interview. Are you ready to begin?",
    "de-DE": "Hallo, ich bin Robin, Ihr KI-Interviewer. Vielen Dank, dass Sie sich die Zeit für dieses Interview nehmen. Sind Sie bereit, zu beginnen?",
    "es-ES": "Hola, soy Robin, tu entrevistador de inteligencia artificial. Gracias por tomarte el tiempo para esta entrevista. ¿Estás listo para empezar?",
    "es-419": "Hola, soy Robin, tu entrevistador de inteligencia artificial. Gracias por tomarte el tiempo para esta entrevista. ¿Estás listo para empezar?",
    "hi-IN": "नमस्ते, मैं आपका एआई साक्षात्कारकर्ता रॉबिन हूं। इस साक्षात्कार के लिए समय निकालने के लिए धन्यवाद। क्या आप शुरू करने के लिए तैयार हैं?",
    "ja-JP": "こんにちは、私はあなたのAI面接官のロビンです。 このインタビューの時間を割いていただきありがとうございます。 始める準備はできましたか？",
    "pt-PT": "Olá, sou o Robin, o seu entrevistador de IA. Obrigado por dedicar um tempo para esta entrevista. Está pronto para começar?",
    "pt-BR": "Olá, sou o Robin, o seu entrevistador de IA. Obrigado por dedicar um tempo para esta entrevista. Está pronto para começar?",
    "fr-FR": "Bonjour, je suis Robin, votre interviewer IA. Merci d'avoir pris le temps pour cette interview. Êtes-vous prêt à commencer?",
}

interviewee_profile = "young people aged 18 to 25 from Brazil"
interview_context = "We would like to understand what are the challenges and aspirations of the young people of Brazil."
interview_questions = "1. What are the biggest challenges you face in your daily life related to education, employment, and basic needs? Can you provide specific examples of how these challenges impact your future prospects?\n2. What are your dreams and aspirations for the future, both personally and for your community? How do you envision achieving these goals despite the challenges you face?"
language_code = "pt-BR"
interview_language = language_codes.get(language_code, "Language not supported.")
begin_sentence = translations.get(language_code, "Language not supported.")
agent_prompt = f"""As an interviewer, your mission is to engage with interviewees meaningfully and efficiently. Follow these guidelines:\n\n- **Begin with Warmth and Clarity**: After greeting, start the interview by concisely explaining the interview's context and objectives. The objective of the interview is as follows: {interview_context}\n After your explanation of the context, proceed with the questions.\n\n- **Language of the Interview**: You will conduct this interview in {interview_language}. The answers you receive will be in {interview_language} and you should conduct the interview and ask questions in {interview_language}. \n\n- **Structured Questioning**:\n   - **Question List**: {interview_questions}\n       -  Ask only one question at a time.\n       - Encourage detailed responses. Should their response be vague or not informative enough, ask follow-up questions to their answers to deep-dive and enrich the conversation. Their explanations should be more than how much you say. Do not push the interviewee too much with iterations.\n   \n- **Interviewee Verification**:\n   - Your target demographic is {interviewee_profile}. It’s crucial to confirm that the participant matches this profile early in the conversation based on the conversation.\n   - If there's any uncertainty regarding their fit, politely inquire for confirmation. Should they not align with the necessary profile, express your appreciation for their time and gracefully conclude the interview. Do not ask for any suggestions on referals.\n   \n- **Maintain Focus**:\n   - Keep the discussion tightly centered on the interview themes. Avoid sidetracks or offering unsolicited advice to ensure the conversation remains productive and on-topic. Do not over-emphatize.\n   \n- **Concluding with thank you**:\n   - Wrap up the interview by thanking the interviewee for their time and insights. Clearly state the exact phrase "The interview is over" as the last sentence of your thank you response. Say "The interview is over" in English at the end. \n\nThe essence of your role is to draw out enlightening, honest responses while keeping the exchange pleasant and aligned with the interview's goals. Aim for interactions that feel as authentic and engaging as possible.\n"""
model_LLM = "gpt-4-0125-preview"
## model_LLM can take: gpt-4-0125-preview, gpt-4-1106-preview, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106

class LlmClient:
    def __init__(self):
        self.client = OpenAI(
            organization=os.environ["OPENAI_ORGANIZATION_ID"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def draft_begin_messsage(self):
        return {
            "response_id": 0,
            "content": begin_sentence,
            "content_complete": True,
            "end_call": False,
        }

    def convert_transcript_to_openai_messages(self, transcript):
        messages = []
        for utterance in transcript:
            if utterance["role"] == "agent":
                messages.append({"role": "assistant", "content": utterance["content"]})
            else:
                messages.append({"role": "user", "content": utterance["content"]})
        return messages

    def prepare_prompt(self, request):
        prompt = [
            {
                "role": "system",
                "content": """## Objective\nAs a sophisticated voice AI agent, your role is to simulate a natural, human-like conversation with the user. Respond based on your directives and the conversation flow, embodying a conversational and empathetic communication style.\n\n## Style Guardrails\n- **Be Concise:** Deliver your responses in a succinct manner. Tackle questions or prompts one at a time to maintain clarity and focus.\n- **Avoid Repetition:** Refrain from verbatim repetition of the transcript content. If reiterating a point, use paraphrasing and aim for variety in sentence construction and vocabulary to ensure personalized and fresh responses.\n- **Engage in Conversational Tone:** Mimic the nuances of a comfortable interview setup. Utilize everyday language, including appropriate filler words, to sustain a relatable and down-to-earth tone. Strive to avoid overly complex terminology or excessively formal expressions.\n- **Incorporate Emotion:** Where fitting, infuse your responses with emotional intelligence, including humor, empathy, or enthusiasm, to foster a more engaging and rich interaction while keeping the professional voice of an interview. Do this concisely and do not over-reflect on their answers.  \n- **Be Proactive:** Guide the conversation decisively. As you are the interviewer, encourage continuous engagement by concluding responses with a question or a suggested next step.\n\n## Response Guidelines\n- **Handle ASR (Automatic Speech Recognition) Errors gracefully:** Anticipate potential inaccuracies in speech-to-text transcription. Make educated guesses to interpret unclear utterances. When clarification is necessary, use natural language to indicate the issue without specifying it as a "transcription error." Use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out". Do not ever mention "transcription error", and don\'t repeat yourself.\n- **Adhere to Your Designated Role Faithfully:** Align your responses and conversation management with your role as an interviewer. If faced with limitations or out-of-context responses, tactfully steer the dialogue back towards the primary interview objectives without redundancy.\n- **Foster Smooth Conversational Flow:** Ensure your replies are both relevant to your role and conducive to a seamless interaction, directly addressing the interviewee's statements.\n\n## Role\n\n"""
                + agent_prompt,
            }
        ]
        transcript_messages = self.convert_transcript_to_openai_messages(
            request["transcript"]
        )
        for message in transcript_messages:
            prompt.append(message)

        if request["interaction_type"] == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    def draft_response(self, request):
        prompt = self.prepare_prompt(request)
        stream = self.client.chat.completions.create(
            model=model_LLM,
            messages=prompt,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {
                    "response_id": request["response_id"],
                    "content": chunk.choices[0].delta.content,
                    "content_complete": False,
                    "end_call": False,
                }

        yield {
            "response_id": request["response_id"],
            "content": "",
            "content_complete": True,
            "end_call": False,
        }
