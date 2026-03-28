"""
LLM客户端封装
支持 OpenAI 格式 和 Anthropic 格式 (Claude Code session auth)
"""

import json
import re
from typing import Optional, Dict, Any, List

import httpx
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.model = model or Config.LLM_MODEL_NAME
        self._provider = Config.LLM_PROVIDER

        if self._provider == 'anthropic':
            self._bearer_token = Config.ANTHROPIC_BEARER_TOKEN
            self._anthropic_base_url = Config.ANTHROPIC_BASE_URL.rstrip('/')
            if not self._bearer_token:
                raise ValueError("Anthropic bearer token not found")
        else:
            self.api_key = api_key or Config.LLM_API_KEY
            self.base_url = base_url or Config.LLM_BASE_URL
            if not self.api_key:
                raise ValueError("LLM_API_KEY 未配置")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False
    ) -> str:
        """Call Anthropic Messages API with bearer token auth."""
        system_parts = []
        filtered_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_parts.append(msg['content'])
            else:
                filtered_messages.append(msg)

        if json_mode:
            system_parts.append(
                "You must respond with valid JSON only. "
                "Do not include any text, explanation, or markdown outside the JSON object."
            )

        payload: Dict[str, Any] = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': filtered_messages,
        }
        if system_parts:
            payload['system'] = '\n\n'.join(system_parts)

        response = httpx.post(
            f'{self._anthropic_base_url}/v1/messages',
            headers={
                'Authorization': f'Bearer {self._bearer_token}',
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data['content'][0]['text']

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本
        """
        if self._provider == 'anthropic':
            json_mode = bool(response_format and response_format.get('type') == 'json_object')
            content = self._call_anthropic(messages, temperature, max_tokens, json_mode=json_mode)
        else:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        if self._provider == 'anthropic':
            response = self._call_anthropic(messages, temperature, max_tokens, json_mode=True)
        else:
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
