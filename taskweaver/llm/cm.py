from typing import Any, Generator, List, Optional

from injector import inject

from taskweaver.llm.base import CompletionService, EmbeddingService, LLMServiceConfig
from taskweaver.llm.util import ChatMessageType, format_chat_message


class CmServiceConfig(LLMServiceConfig):
    def _configure(self) -> None:
        self._set_name("cm")

        self.api_base = self.llm_module_config.api_base

        shared_api_key = self.llm_module_config.api_key
        self.api_key = self._get_str(
            "api_key",
            shared_api_key,
        )

        shared_model = self.llm_module_config.model
        self.model = self._get_str(
            "model",
            shared_model if shared_model is not None else "pretrain/Qwen1.5-72B-Chat",
        )

        shared_embedding_model = self.llm_module_config.embedding_model
        self.embedding_model = self._get_str(
            "embedding_model",
            (
                shared_embedding_model
                if shared_embedding_model is not None
                else self.model
            ),
        )


class CmService(CompletionService, EmbeddingService):
    httpx = None
    httpx_sse = None

    @inject
    def __init__(self, config: CmServiceConfig):
        self.config = config

        if CmService.httpx is None:
            try:
                import httpx
                import httpx_sse

                CmService.httpx = httpx
                CmService.httpx_sse = httpx_sse
            except Exception:
                raise Exception(
                    "Package httpx/httpx_sse is required for using CM API",
                )

    def chat_completion(
        self,
        messages: List[ChatMessageType],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageType, None, None]:
        engine = self.config.model
        base_url = self.config.api_base

        import json

        role: Any = None
        content: str = None

        history = []
        prompt, history = self._messages_to_prompt_n_history(messages)

        data = {"prompt": prompt, "history": history, "stream": stream}

        print("======== REQUEST ========")
        print(data)

        with CmService.httpx.Client(timeout=120) as client:
            with CmService.httpx_sse.connect_sse(
                client, "POST", base_url, json=data
            ) as event_source:
                for stream_res in event_source.iter_sse():
                    try:
                        data = json.loads(stream_res.data)
                    except:
                        continue

                    delta = data["delta"]

                    if delta == "[EOS]":
                        print("======== RESPONSE ========")
                        print(data["response"])
                        continue

                    yield format_chat_message(role, delta)

        # data = {"model": engine, "messages": messages, "stream": stream}

        # if stream:
        #     with CmService.httpx.Client() as client:
        #         with CmService.httpx_sse.connect_sse(client, "POST", "http://ip:port/v1/chat/completions", json=data) as event_source:
        #             for stream_res in event_source.iter_sse():
        #                 if not stream_res.data:
        #                     continue
        #                 print("======== RESPOND ========")
        #                 print(stream_res.data)
        #                 try:
        #                     data = json.loads(stream_res.data)
        #                 except:
        #                     continue
        #                 delta = data['choices'][0]['delta']
        #                 if delta is None:
        #                     continue
        #                 if 'role' in delta:
        #                     role = delta['role']
        #                     # print("-role-", role)
        #                     continue
        #                 if 'content' in delta:
        #                     content = delta['content']
        #                     # print("-content-", content)cd
        #                     if content is None:
        #                         continue
        #                 yield format_chat_message(role, content)
        # else:
        #     with CmService.httpx.Client() as client:
        #         res = client.post("http://ip:port/v1/chat/completions", body=data)
        #         response = json.loads(res.choices[0].message)
        #         yield response

    def _messages_to_prompt_n_history(
        self, messages: List[ChatMessageType]
    ) -> tuple[str, List[str]]:

        print(messages)

        history = []
        prompt = messages.pop()["content"]

        if messages:
            if messages[0]["role"] == "system":
                system_prompt = messages.pop(0)["content"]
                # prompt = system_prompt + "\n" + prompt

            for message in messages:
                if message["role"] == "user":
                    user_content = message["content"]
                else:
                    round = [user_content, message["content"]]
                    history.append(round)

        return prompt, history

    def get_embeddings(self, strings: List[str]) -> List[List[float]]:
        pass
