import asyncio
import websockets
import json
import logging
from datetime import datetime
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsServer:
    def __init__(self):
        self.clients = set()
        self.metrics = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_memory_used': 0,
            'throughput': 0
        }

    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def update_metrics(self):
        """Update system metrics"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                self.metrics.update({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Broadcast to all clients
                if self.clients:
                    message = json.dumps(self.metrics)
                    await asyncio.gather(
                        *[client.send(message) for client in self.clients]
                    )
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(5)

    async def start_server(self):
        """Start the WebSocket server"""
        async with websockets.serve(self.register, "localhost", 8765):
            await self.update_metrics()

def main():
    """Main entry point"""
    server = MetricsServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()