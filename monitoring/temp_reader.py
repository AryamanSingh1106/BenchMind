import logging
import requests

logger = logging.getLogger("BenchMind.TempReader")


def clean_temp(value):
    """
    Convert '62.0 °C' → 62.0
    """
    if value is None:
        return None

    try:
        # remove °C and spaces
        value_str = str(value).replace("°C", "").strip()
        return float(value_str)
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug("Failed to clean temperature value '%s': %s", value, e)
        return None


def get_temperatures():
    temps = {
        "cpu_temp": None,
        "gpu_temp": None,
    }

    try:
        response = requests.get(
            "http://localhost:8085/data.json",
            timeout=1
        )
        response.raise_for_status()
        data = response.json()

        def scan(node):
            if isinstance(node, dict):
                if "Children" in node and isinstance(node["Children"], list):
                    for child in node["Children"]:
                        scan(child)

                if node.get("Type") == "Temperature":
                    name = node.get("Text", "").lower()
                    if "cpu package" in name:
                        temps["cpu_temp"] = clean_temp(node.get("Value"))
                    if "gpu core" in name:
                        temps["gpu_temp"] = clean_temp(node.get("Value"))

        scan(data)
        return temps

    except requests.RequestException as e:
        logger.debug("LibreHardwareMonitor web server is unavailable: %s", e)
        return temps
    except Exception as e:
        logger.warning("Unexpected error reading hardware temperatures: %s", e, exc_info=True)
        return temps