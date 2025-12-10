import httpx

from client.errors import handle_transport_errors


class Transport:
    def __init__(self, base_url: str, timeout: float = 60) -> None:
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    @handle_transport_errors
    def request(self, method: str, path: str, **kwargs):
        response = self.client.request(method=method, url=path, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get(self, path: str, **kwargs): 
        return self.request("GET", path, **kwargs)
    
    def post(self, path: str, **kwargs): 
        return self.request("POST", path, **kwargs)
    
    def patch(self, path: str, **kwargs): 
        return self.request("PATCH", path, **kwargs)
    
    def delete(self, path: str, **kwargs): 
        return self.request("DELETE", path, **kwargs)
    
    def close(self) -> None:
        self.client.close()