"""Shared in-process bridge for QQ Official group lifecycle events.

AstrBot v4.26 does not forward GROUP_DEL_ROBOT into its event bus, while
qq-botpy already parses it. This runtime patch does not modify AstrBot files.
"""
from __future__ import annotations

import builtins
import inspect
import logging
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("astrbot")
_STATE_KEY = "_ASTRBOT_QQ_GROUP_LIFECYCLE_BRIDGE_V1"
Callback = Callable[[Any, Any], Awaitable[None]]


def _state() -> dict[str, Any]:
    state = getattr(builtins, _STATE_KEY, None)
    if not isinstance(state, dict):
        state = {"installed": False, "callbacks": {}}
        setattr(builtins, _STATE_KEY, state)
    return state


def install(owner: str, callback: Callback) -> None:
    state = _state()
    state["callbacks"][owner] = weakref.WeakMethod(callback)
    if state["installed"]:
        return

    from astrbot.core.platform.sources.qqofficial import qqofficial_platform_adapter as module

    original = getattr(module.botClient, "on_group_del_robot", None)

    async def patched(client, event):
        if original is not None:
            result = original(client, event)
            if inspect.isawaitable(result):
                await result
        await _dispatch(client, event)

    module.botClient.on_group_del_robot = patched
    state["installed"] = True
    state["original"] = original
    logger.warning("[QQGroupLifecycle] GROUP_DEL_ROBOT runtime bridge installed")


def detach(owner: str) -> None:
    _state()["callbacks"].pop(owner, None)


async def _dispatch(client: Any, event: Any) -> None:
    state = _state()
    stale = []
    for owner, callback_ref in list(state["callbacks"].items()):
        callback = callback_ref() if callback_ref else None
        if callback is None:
            stale.append(owner)
            continue
        try:
            await callback(client, event)
        except Exception:
            logger.exception("[QQGroupLifecycle] callback failed owner=%s", owner)
    for owner in stale:
        state["callbacks"].pop(owner, None)
