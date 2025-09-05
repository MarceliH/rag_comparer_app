from mistralai import Mistral

def check_mistralai_connection(api_key):
    try:
        client = Mistral(api_key=api_key)
        # Try to list models to verify connection
        _ = client.models.list()
        return True
    except Exception:
        return False