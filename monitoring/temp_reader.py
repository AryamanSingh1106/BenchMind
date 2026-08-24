import requests


def clean_temp(value):
    """
    Convert '62.0 °C' → 62.0
    """
    try:
        if value is None:
            return None

        # remove °C and spaces
        value = str(value).replace("°C", "").strip()

        return float(value)

    except:
        return None


def get_temperatures():

    try:
        data = requests.get(
            "http://localhost:8085/data.json",
            timeout=1
        ).json()

        temps = {
            "cpu_temp": None,
            "gpu_temp": None,
        }

        def scan(node):

            if "Children" in node:
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

    except:
        return {
            "cpu_temp": None,
            "gpu_temp": None
        }