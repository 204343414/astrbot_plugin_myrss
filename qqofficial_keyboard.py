"""QQ Official Keyboard interaction bridge for AstrBot v4.26.x.

Narrow compatibility patch:
- enable botpy interaction intent before platform initialization;
- dispatch INTERACTION_CREATE to registered plugin handlers;
- ACK every interaction exactly once.

Remove this bridge when AstrBot exposes interaction events natively.
"""

import builtins
import logging

logger = logging.getLogger("astrbot")

REGISTRY_NAME = "_ASTRBOT_QQ_INTERACTION_HANDLERS"
PATCH_FLAG = "_ASTRBOT_QQ_INTERACTION_PATCHED"


def _registry() -> dict:
    if not hasattr(builtins, REGISTRY_NAME):
        setattr(builtins, REGISTRY_NAME, {})
    return getattr(builtins, REGISTRY_NAME)


def register_handler(name: str, handler) -> None:
    _registry()[name] = handler


def unregister_handler(name: str) -> None:
    _registry().pop(name, None)


def install_patch() -> None:
    if getattr(builtins, PATCH_FLAG, False):
        return
    from astrbot.core.platform.sources.qqofficial import qqofficial_platform_adapter as module

    original_init = module.QQOfficialPlatformAdapter.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Plugins are instantiated before platform adapters during a full AstrBot start.
        self.intents.interaction = True
        self.client.intents.interaction = True
        logger.info("[QQKeyboard] interaction intent enabled for platform %s", self.config.get("id"))

    async def on_interaction_create(client, interaction):
        code = 1  # operation failed unless a handler claims it
        claimed = False
        for name, handler in list(_registry().items()):
            try:
                result = await handler(client, interaction)
                if result is not None:
                    code = int(result)
                    claimed = True
                    break
            except Exception as exc:
                logger.exception("[QQKeyboard] handler %s failed: %s", name, exc)
                code = 1
                claimed = True
                break
        try:
            await client.api.on_interaction_result(interaction.id, code if claimed else 1)
        except Exception as exc:
            logger.error("[QQKeyboard] interaction ACK failed: %s", exc)

    module.QQOfficialPlatformAdapter.__init__ = patched_init
    module.botClient.on_interaction_create = on_interaction_create
    setattr(builtins, PATCH_FLAG, True)
    logger.warning("[QQKeyboard] AstrBot v4.26 compatibility patch installed; full restart required")
