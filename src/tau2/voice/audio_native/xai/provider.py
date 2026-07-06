"""
xAI Grok Voice Agent API provider for real-time voice processing.

Uses WebSocket for bidirectional audio streaming with the Grok Voice Agent API.
The API is very similar to OpenAI's Realtime API with minor differences in event names.

Key advantages:
- Native G.711 μ-law support (no audio conversion for telephony!)
- OpenAI-compatible protocol
- Built-in server VAD

Reference: https://docs.x.ai/docs/guides/voice/agent
"""

import asyncio
import base64
import json
import os
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional
from urllib.parse import urlencode

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_TELEPHONY_RATE,
    DEFAULT_XAI_MODEL,
    DEFAULT_XAI_REALTIME_BASE_URL,
    DEFAULT_XAI_VOICE,
)
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.xai.events import (
    BaseXAIEvent,
    XAITimeoutEvent,
    XAIUnknownEvent,
    parse_xai_event,
)

load_dotenv()


class XAIAudioFormat(str, Enum):
    """Audio formats supported by xAI Grok Voice Agent API."""

    # G.711 μ-law at 8kHz - optimal for telephony (no conversion needed!)
    PCMU = "audio/pcmu"
    # G.711 A-law at 8kHz - international telephony
    PCMA = "audio/pcma"
    # PCM Linear16 at configurable sample rate (default 24kHz)
    PCM = "audio/pcm"


class XAIVADMode(str, Enum):
    """Voice Activity Detection modes for xAI Grok Voice Agent API."""

    SERVER_VAD = "server_vad"  # Server handles VAD automatically
    MANUAL = "manual"  # Client controls turns explicitly (turn_detection: null)


class XAIVADConfig(BaseModel):
    """Configuration for xAI's Voice Activity Detection.

    Attributes:
        mode: VAD mode. Defaults to SERVER_VAD.
        threshold: Speech detection sensitivity. xAI's documented default is
            server-controlled when omitted.
        prefix_padding_ms: Audio to include before speech start.
        silence_duration_ms: Silence before turn end.
        idle_timeout_ms: Server-side idle timeout before an empty audio commit.
    """

    mode: XAIVADMode = XAIVADMode.SERVER_VAD
    threshold: Optional[float] = None
    prefix_padding_ms: Optional[int] = None
    silence_duration_ms: Optional[int] = None
    idle_timeout_ms: Optional[int] = None


class XAIRealtimeProvider:
    """xAI Grok Voice Agent API provider with WebSocket-based communication.

    This provider manages a persistent WebSocket connection to xAI's Realtime API,
    enabling real-time bidirectional communication for voice and text processing.

    Key advantages over OpenAI/Gemini:
    - Native G.711 μ-law support (no audio conversion for telephony!)
    - OpenAI-compatible API (similar event names and protocol)

    Attributes:
        BASE_URL: The WebSocket endpoint for xAI's Realtime API.
        api_key: The xAI API key for authentication.
        voice: The voice to use (Ara, Rex, Sal, Eve, Leo).
        ws: The active WebSocket connection, or None if disconnected.

    Example:
        ```python
        provider = XAIRealtimeProvider()
        await provider.connect()
        await provider.configure_session(
            system_prompt="You are a helpful assistant.",
            tools=[],
            vad_config=XAIVADConfig(),
        )
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    BASE_URL = DEFAULT_XAI_REALTIME_BASE_URL
    DEFAULT_VOICE = DEFAULT_XAI_VOICE
    DEFAULT_MODEL = DEFAULT_XAI_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: XAIAudioFormat = XAIAudioFormat.PCMU,
        sample_rate: int = DEFAULT_TELEPHONY_RATE,
    ):
        """Initialize the xAI Realtime provider.

        Args:
            api_key: xAI API key. If not provided, reads from XAI_API_KEY
                environment variable.
            model: Voice model to use. Defaults to grok-voice-think-fast-1.0.
            reasoning_effort: Optional reasoning effort query parameter.
                xAI supports "high" and "none"; None uses the server default.
            voice: Voice to use. One of: Ara, Rex, Sal, Eve, Leo. Defaults to Ara.
            audio_format: Audio format for input/output. Defaults to PCMU (G.711 μ-law)
                which is optimal for telephony (no conversion needed).
            sample_rate: Sample rate for PCM format (ignored for PCMU/PCMA).
                Supported: 8000, 16000, 24000, 32000, 44100, 48000. Default: 8000.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key not provided. Set XAI_API_KEY env var.")

        if reasoning_effort not in {None, "high", "none"}:
            raise ValueError(
                "xAI provider supports reasoning_effort='high', 'none', or None "
                f"(got {reasoning_effort!r})"
            )

        self.model = model or self.DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort
        self.voice = voice or self.DEFAULT_VOICE
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[XAIVADConfig] = None
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    def _build_connect_url(self) -> str:
        query = {"model": self.model}
        if self.reasoning_effort is not None:
            query["reasoning.effort"] = self.reasoning_effort
        return f"{self.BASE_URL}?{urlencode(query)}"

    @websocket_retry
    async def connect(self) -> None:
        """Establish a WebSocket connection to the xAI Realtime API.

        Opens a new WebSocket connection and waits for the initial session event
        event to confirm successful connection.

        Raises:
            RuntimeError: If the initial handshake fails.
        """
        if self.is_connected:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = self._build_connect_url()
        logger.info(f"xAI Realtime API: Connecting to {url}")
        self.ws = await websockets.connect(url, additional_headers=headers)

        # Wait for initial session event. xAI previously emitted
        # conversation.created here and now emits session.created.
        response = await self.ws.recv()
        data = json.loads(response)
        logger.debug(f"xAI Realtime API: handshake event payload: {response}")
        event_type = data.get("type")
        if event_type not in {"conversation.created", "session.created"}:
            raise RuntimeError(
                "Expected conversation.created or session.created, "
                f"got {event_type}"
            )

        # Store conversation/session ID for debugging.
        session_data = data.get("session") or data.get("conversation") or {}
        self.session_id = session_data.get("id") or data.get("event_id")
        logger.info(
            f"xAI Realtime API: Connected successfully (session_id={self.session_id})"
        )

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            logger.info("xAI Realtime API: Disconnecting")
            await self.ws.close()
            self.ws = None
            logger.info("xAI Realtime API: Disconnected")

    def _build_audio_config(self) -> Dict:
        """Build the audio configuration for the session."""
        if self.audio_format == XAIAudioFormat.PCMU:
            # G.711 μ-law at 8kHz (telephony)
            return {
                "input": {"format": {"type": "audio/pcmu"}},
                "output": {"format": {"type": "audio/pcmu"}},
            }
        elif self.audio_format == XAIAudioFormat.PCMA:
            # G.711 A-law at 8kHz
            return {
                "input": {"format": {"type": "audio/pcma"}},
                "output": {"format": {"type": "audio/pcma"}},
            }
        else:
            # PCM with configurable sample rate
            return {
                "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                "output": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
            }

    def _build_turn_detection_config(self, vad_config: XAIVADConfig) -> Optional[Dict]:
        """Build the turn detection configuration."""
        if vad_config.mode == XAIVADMode.MANUAL:
            return {"type": None}
        config: Dict[str, object] = {"type": "server_vad"}
        if vad_config.threshold is not None:
            config["threshold"] = vad_config.threshold
        if vad_config.prefix_padding_ms is not None:
            config["prefix_padding_ms"] = vad_config.prefix_padding_ms
        if vad_config.silence_duration_ms is not None:
            config["silence_duration_ms"] = vad_config.silence_duration_ms
        if vad_config.idle_timeout_ms is not None:
            config["idle_timeout_ms"] = vad_config.idle_timeout_ms
        return config

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for the xAI API.

        xAI uses the same tool format as OpenAI.
        """
        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            formatted_tools.append(
                {
                    "type": "function",
                    "name": schema["function"]["name"],
                    "description": schema["function"]["description"],
                    "parameters": schema["function"]["parameters"],
                }
            )
        return formatted_tools

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: XAIVADConfig,
    ) -> None:
        """Configure the realtime session with instructions, tools, and settings.

        Args:
            system_prompt: The system instructions for the assistant.
            tools: List of tools available for the assistant to use.
            vad_config: Voice Activity Detection configuration.

        Raises:
            RuntimeError: If not connected or if session configuration fails.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        session_config = {
            "type": "session.update",
            "session": {
                "voice": self.voice,
                "instructions": system_prompt,
                "audio": self._build_audio_config(),
                "turn_detection": self._build_turn_detection_config(vad_config),
                "tools": self._format_tools_for_api(tools),
            },
        }

        logger.debug(f"xAI session config: {json.dumps(session_config, indent=2)}")
        await self.ws.send(json.dumps(session_config))

        # Wait for session.updated confirmation
        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type == "session.updated":
                self._current_vad_config = vad_config
                logger.info("xAI Realtime API: Session configured successfully")
                logger.debug(f"xAI Realtime API: session.updated payload: {response}")
                break
            elif event_type == "error":
                error = data.get("error", {})
                error_msg = error.get("message", str(error))
                raise RuntimeError(f"Session configuration failed: {error_msg}")

    async def send_audio(self, audio_data: bytes) -> None:
        """Append audio data to the input audio buffer.

        Audio should be in the configured format (G.711 μ-law by default).

        Args:
            audio_data: Raw audio bytes.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        gain_env = os.environ.get("TAU2_XAI_INPUT_GAIN")
        if gain_env and self.audio_format == XAIAudioFormat.PCMU:
            import audioop

            gain = float(gain_env)
            if gain != 1.0:
                lin = audioop.ulaw2lin(audio_data, 2)
                lin = audioop.mul(lin, 2, gain)
                audio_data = audioop.lin2ulaw(lin, 2)

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.ws.send(json.dumps(message))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send the result of a tool/function call back to the API.

        Args:
            call_id: The unique identifier of the function call.
            result: The string result of the function execution.
            request_response: If True, request the assistant to continue.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self.ws.send(json.dumps(item_create))

        if request_response:
            await self.create_response()

    async def send_synthetic_agent_context(
        self, content: str, request_response: bool = True
    ) -> None:
        """Send custom agent-side text context as a synthetic conversation item."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            },
        }
        await self.ws.send(json.dumps(item_create))

        if request_response:
            await self.create_response()

    async def create_response(self) -> None:
        """Ask xAI to produce the next assistant response."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Tell the server how much assistant audio was played."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        }
        await self.ws.send(json.dumps(truncate_event))
        logger.debug(
            f"xAI truncate sent: item_id={item_id}, content_index={content_index}, "
            f"audio_end_ms={audio_end_ms}"
        )

    async def receive_events(self) -> AsyncGenerator[BaseXAIEvent, None]:
        """Receive and yield events from the WebSocket connection.

        Yields:
            BaseXAIEvent: Parsed event objects.

        Raises:
            RuntimeError: If connection closes unexpectedly.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                event = parse_xai_event(data)
                yield event

            except asyncio.TimeoutError:
                yield XAITimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"xAI Realtime API: WebSocket connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                ) from e
            except websockets.ConnectionClosedError as e:
                logger.error(
                    f"xAI Realtime API: WebSocket connection error "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                ) from e
            except Exception as e:
                logger.error(f"xAI Realtime API: Error receiving event: {e}")
                yield XAIUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseXAIEvent]:
        """Receive events for a specified duration.

        Collects all events that arrive within the specified time window.
        Useful for tick-based processing.

        Args:
            duration_seconds: How long to collect events.

        Returns:
            List of events received during the duration.
        """
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, XAITimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
