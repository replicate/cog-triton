import asyncio
import dataclasses
import json
import logging
import time
from typing import Callable, AsyncIterator

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
)

log = logging.getLogger("webrtc")


class ShutdownTimer(asyncio.Event):
    def __init__(self, timeout: int = 5) -> None:
        self.deadline = time.monotonic() + timeout
        self.timeout = timeout
        self.task = asyncio.create_task(self.exit())
        super().__init__()

    def reset(self) -> None:
        self.deadline = time.monotonic() + self.timeout

    async def exit(self) -> None:
        while not self.is_set():
            await asyncio.sleep(self.deadline - time.monotonic())
            if self.deadline < time.monotonic():
                log.info("ping deadline exceeded")
                self.set()


@dataclasses.dataclass
class RTC:
    offer: str

    def on_message(self, f: Callable[[dict], AsyncIterator[dict]]) -> None:
        self.wrapped_message_handler = f

    async def message_handler(
        self, message: bytes | str
    ) -> AsyncIterator[bytes | str | dict]:
        if message[0] != "{":
            log.info("received invalid message", message)
            return
        args = json.loads(message)  # works for bytes or str
        id = args.pop("id", 0)
        async for result in self.wrapped_message_handler(args):
            result["id"] = id
            yield result
            # yield json.dumps(result)

    # def serve_with_loop(self, loop: asyncio.AbstractEventLoop) -> Iterator[str]:
    #     """
    #     this is so that you can do `yield from rtc.serve_with_loop(loop)`

    #     you can't `yield from` in an async function, so in that case the caller
    #     would need to do `yield await rtc.answer(); yield await rtc.wait_disconnect()`
    #     """
    #     yield loop.run_until_complete(self.answer())
    #     yield loop.run_until_complete(self.wait_disconnect())

    async def answer(self) -> str:
        log.info("handling offer")
        params = json.loads(self.offer)
        ice_servers = params.get("ice_servers", "[]")

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        log.info("creating for", offer)
        config = RTCConfiguration([RTCIceServer(**a) for a in ice_servers])
        log.info("configured for", ice_servers)
        pc = RTCPeerConnection(configuration=config)
        log.info("made peerconnection", pc)

        # five seconds to establish a connection and ping!
        self.done = ShutdownTimer()
        self.token_stream = asyncio.Queue()

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            log.info(type(channel))

            @channel.on("message")
            async def on_message(message: str | bytes) -> None:
                log.info(message)
                if isinstance(message, str) and message.startswith("ping"):
                    # recepient can use our time + rt ping latency to estimate clock drift
                    # if they send time as the ping message and record received time,
                    # drift = (their time) - ((time we sent) + (roundtrip latency) / 2)
                    channel.send(f"pong{message[4:]} {round(time.time() * 1000)}")
                    self.done.reset()
                else:
                    async for result in self.message_handler(message):
                        channel.send(json.dumps(result))
                        self.token_stream.put_nowait(result)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            log.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.done.set()

        # handle offer
        await pc.setRemoteDescription(offer)
        log.info("set remote description")

        # send answer
        answer = await pc.createAnswer()
        log.info("created answer", answer)
        await pc.setLocalDescription(answer)
        log.info("set local description")
        data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        return json.dumps(data)

    # async def wait_disconnect(self) -> str:
    #     await self.done.wait()
    #     return "disconnected"

    async def yield_output(self) -> AsyncIterator:
        yield await self.answer()
        while True:
            get_task = asyncio.create_task(self.token_stream.get())
            wait_task = asyncio.create_task(self.done.wait())
            (result,), pending = await asyncio.wait(
                [wait_task, get_task], return_when=asyncio.FIRST_COMPLETED
            )
            if self.done.is_set():
                yield "disconnected"
                get_task.cancel()
                return
            yield result
