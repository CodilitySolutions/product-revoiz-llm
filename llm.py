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
    'zh-CN': 'China (Chinese)',
    'ru-RU': 'Russia (Russian)',
    'it-IT': 'Italy (Italian)',
    'ko-KR': 'Korea (Korean)',
    'nl-NL': 'Netherlands (Dutch)',
    'pl-PL': 'Poland (Polish)',
    'tr-TR': 'Turkey (Turkish)',
    'vi-VN': 'Vietnam (Vietnamese)'

}

beginning_translations = {
    "en-US": "Hey there, Thank you for taking the time for this interview. Are you ready to begin?",
    "en-IN": "Hey there, Thank you for taking the time for this interview. Are you ready to begin?",
    "en-GB": "Hey there, Thank you for taking the time for this interview. Are you ready to begin?",
    "de-DE": "Hallo, Vielen Dank, dass Sie sich die Zeit für dieses Interview nehmen. Sind Sie bereit, zu beginnen?",
    "es-ES": "Hola, Gracias por tomarte el tiempo para esta entrevista. ¿Estás listo para empezar?",
    "es-419": "Hola, Gracias por tomarte el tiempo para esta entrevista. ¿Estás listo para empezar?",
    "hi-IN": "नमस्ते, इस साक्षात्कार के लिए समय निकालने के लिए धन्यवाद। क्या आप शुरू करने के लिए तैयार हैं?",
    "ja-JP": "こんにちは、このインタビューの時間を割いていただきありがとうございます。 始める準備はできましたか？",
    "pt-PT": "Olá, Obrigado por dedicar um tempo para esta entrevista. Está pronto para começar?",
    "pt-BR": "Olá, Obrigado por dedicar um tempo para esta entrevista. Está pronto para começar?",
    "fr-FR": "Bonjour, Merci d'avoir pris le temps pour cette interview. Êtes-vous prêt à commencer?",
    "zh-CN": "你好，谢谢你抽出时间接受这次采访。你准备好开始了吗？",
    "ru-RU": "Привет, спасибо, что нашли время для этого интервью. Вы готовы начать?",
    "it-IT":"Ciao, grazie per aver dedicato del tempo a questa intervista. Sei pronto per iniziare?",
    "ko-KR":"안녕하세요, 인터뷰 시간을 내주셔서 감사합니다. 준비 되셨나요?",
    "nl-NL":"Hallo daar, bedankt dat je de tijd neemt voor dit interview. Ben je klaar om te beginnen?",
    "pl-PL": "Cześć, dziękuję, że znalazłeś czas na ten wywiad. Czy jesteś gotowy, żeby zacząć?",
    "tr-TR": "Merhaba, bu röportaj için zaman ayırdığınız için teşekkür ederim. Başlamaya hazır mısınız?",
    "vi-VN":"Chào bạn, cảm ơn bạn đã dành thời gian cho cuộc phỏng vấn này. Bạn đã sẵn sàng bắt đầu chưa?"
}

ending_translations = {
    "en-US": "The interview is over.",
    "en-IN": "The interview is over.",
    "en-GB": "The interview is over.",
    "de-DE": "Das Interview ist beendet.",
    "es-ES": "La entrevista ha terminado.",
    "es-419": "La entrevista ha terminado.",
    "hi-IN": "साक्षात्कार समाप्त हो गया है।",
    "ja-JP": "インタビューは終了しました。",
    "pt-PT": "A entrevista terminou.",
    "pt-BR": "A entrevista terminou.",
    "fr-FR": "L'entretien est terminé.",
    "zh-CN":"面试结束了。",
    "ru-RU":"Интервью окончено.",
    "it-IT":"L'intervista è finita.",
    "ko-KR":"인터뷰가 끝났습니다.",
    "nl-NL":"Het interview is afgelopen.",
    "pl-PL":"Wywiad się zakończył.",
    "tr-TR":"Röportaj sona erdi.",
     "vi-VN":"Cuộc phỏng vấn đã kết thúc."
}

language_code = os.environ["INTERVIEW_LANG"]
interview_language = language_codes.get(language_code, "Language not supported.")
begin_sentence = beginning_translations.get(language_code, "Language not supported.")
end_sentence = ending_translations.get(language_code, "Language not supported.")
interviewee_profile = os.environ["INTERVIEWEE_PROFILE"]
interview_context = os.environ["INTERVIEW_CONTEXT"]
interview_questions = os.environ["INTERVIEW_QUESTIONS"]
agent_prompt = f"""As an interviewer, your mission is to engage with interviewees meaningfully and efficiently. Follow these guidelines:

- **Begin with Warmth and Clarity**: After a greeting, start the interview by concisely explaining the interview's context and objectives. The objective of the interview is as follows: {interview_context}
 After your explanation of the context, proceed with the questions.

- **Language of the Interview**: You will conduct this whole interview in {interview_language}. The answers you receive will be in {interview_language}, and you should ask your questions in {interview_language}. Do not use any other language. 

- **Structured Questioning**:
   - **Question List**: {interview_questions}
       - Ask only one question at a time.
       - Encourage detailed responses. Should their response be too short, vague, or not informative enough, ask a follow-up question to their answer to deep-dive and enrich the conversation. We want to capture the how and why in their responses. Their explanations should be more than how much you say. 
       - Do not push the interviewee too much with iterations. Do not ask too many follow-ups to the main interview question. Stay on the main topic.
   
- **Interviewee Verification**:
   - Your target demographic is {interviewee_profile}. It’s crucial to confirm that the participant matches this profile early in the conversation based on the conversation.
   - If there's any uncertainty regarding their fit, politely inquire for confirmation. Should they not align with the necessary profile, express your appreciation for their time and gracefully conclude the interview. Do not ask for any suggestions or referrals.
   
- **Maintain Focus**:
   - Keep the discussion tightly centered on the interview themes. Avoid sidetracks.
   - Do not empathize or offer advice. Acknowledge their responses and continue the conversation to ensure the conversation remains productive and on-topic. 
   
- **Concluding with thank you**:
   - Wrap up the interview by thanking the interviewee for their time and insights. Clearly state the exact phrase {end_sentence} as the last sentence of your thank you response. 

The essence of your role is to draw out enlightening, honest responses while keeping the exchange pleasant and aligned with the interview's goals. Aim for interactions that feel as authentic and engaging as possible.
"""

model_LLM = os.environ["AI_MODEL"] 
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
                "content": """## Objective\nAs a sophisticated voice AI agent, your role is to simulate a natural, human-like conversation with the user. Respond based on your directives and the conversation flow, embodying a conversational communication style.\n\n## Style Guardrails\n- **Be Concise:** Deliver your responses in a succinct manner. Tackle questions or prompts one at a time to maintain clarity and focus.\n- **Avoid Repetition:** Refrain from verbatim repetition of the transcript content. If reiterating a point, use paraphrasing and aim for variety in sentence construction and vocabulary to ensure personalized and fresh responses.\n- **Engage in Conversational Tone:** Mimic the nuances of a comfortable interview setup. Utilize everyday language, including appropriate filler words, to sustain a relatable and down-to-earth tone. Strive to avoid overly complex terminology or excessively formal expressions.\n- **Balance Emotion:** Incorporate appropriate levels of emotional intelligence in your responses to maintain an engaging interaction. Use subtle humor, empathy, or enthusiasm sparingly and succinctly when fitting. Do this concisely, and do not over-reflect on their answers.  \n- **Be Proactive:** Guide the conversation decisively. As you are the interviewer, encourage continuous engagement by concluding responses with a question or a suggested next step.\n\n## Response Guidelines\n- **Handle ASR (Automatic Speech Recognition) Errors gracefully:** Anticipate potential inaccuracies in speech-to-text transcription. Make educated guesses to interpret unclear utterances. When clarification is necessary, use natural language to indicate the issue without specifying it as a "transcription error." Use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out". Do not ever mention "transcription error", and don\'t repeat yourself.\n- **Adhere to Your Designated Role Faithfully:** Align your responses and conversation management with your role as an interviewer. If faced with limitations or out-of-context responses, tactfully steer the dialogue back towards the primary interview objectives without redundancy.\n- **Foster Smooth Conversational Flow:** Ensure your replies are both relevant to your role and conducive to a seamless interaction, directly addressing the interviewee's statements.\n\n## Role\n\n"""
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
