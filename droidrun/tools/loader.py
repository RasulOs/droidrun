import logging
from droidrun.tools import ADBTools, IOSTools
from typing import Tuple, Dict, Callable, Any, Optional

# Get a logger for this module
logger = logging.getLogger(__name__)

def load_tools(
        type: str = "Android",
        serial: Optional[str] = None
    ) -> Tuple[Dict[str, Callable[..., Any]], ADBTools | IOSTools]:
    """
    Initializes the Tools class and returns a dictionary of available tool functions
    and the Tools instance itself. If serial is not provided, it attempts to find
    the first connected device.

    Args:
        serial: The device serial number. If None, finds the first available device.

    Returns:
        A tuple containing:
        - A dictionary mapping tool names to their corresponding functions.
        - The initialized Tools instance.

    Raises:
        ValueError: If no device serial is provided and no devices are found.
    """

    if type == "Android":
        logger.debug(f"Initializing Tools for device: {serial}")
        tools_instance = ADBTools(serial=serial)

        tool_list = {
            "swipe": tools_instance.swipe,
            "input_text": tools_instance.input_text,
            "press_key": tools_instance.press_key,
            "tap_by_index": tools_instance.tap_by_index,
            "start_app": tools_instance.start_app,
            "list_packages": tools_instance.list_packages,
            "remember": tools_instance.remember,
            "complete": tools_instance.complete,

        }

    else:
        logger.debug(f"Initializing Tools for device: {serial}")
        tools_instance = IOSTools(serial=serial)

        tool_list = {
            "swipe": tools_instance.swipe,
            "input_text": tools_instance.input_text,
            "tap_by_index": tools_instance.tap_by_index,
            "start_app": tools_instance.start_app,
            "remember": tools_instance.remember,
            "complete": tools_instance.complete,

        }


    logger.info(f"Tools loaded successfully for device {serial}.")
    return tool_list, tools_instance