class MetricsHandler {
  constructor() {
    this.ws = null;
    this.metrics = {
      cpu_percent: 0,
      memory_percent: 0,
      gpu_memory_used: 0,
      throughput: 0,
      avg_inference_time: 0,
      avg_batch_size: 32,
      memory_efficiency: 0,
      gpu_utilization: 0
    };
    this.listeners = new Set();
  }

  connect() {
    this.ws = new WebSocket('ws://localhost:8765/metrics');
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.metrics = { ...this.metrics, ...data };
      this.notifyListeners();
    };

    this.ws.onclose = () => {
      setTimeout(() => this.connect(), 1000);
    };
  }

  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notifyListeners() {
    this.listeners.forEach(callback => callback(this.metrics));
  }
}

const metricsHandler = new MetricsHandler();
export default metricsHandler;