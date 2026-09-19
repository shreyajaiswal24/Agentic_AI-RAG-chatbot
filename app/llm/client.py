"""Chat model factory. Both providers speak the same LangChain interface, so the
graph never knows which one is in use."""

from langchain_core.language_models import BaseChatModel

from app.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=settings.llm_temperature)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=settings.llm_temperature)
