import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Cpu, Database, TrendingUp } from 'lucide-react';

const MonitoringDashboard = ({ metrics }) => {
  const [historicalData, setHistoricalData] = useState([]);
  const [selectedMetric, setSelectedMetric] = useState('cpu');

  useEffect(() => {
    if (metrics) {
      setHistoricalData(prev => [...prev, {
        timestamp: new Date().getTime(),
        cpu: metrics.cpu_percent,
        memory: metrics.memory_percent,
        gpu: metrics.gpu_memory_used / 1e9, // Convert to GB
        throughput: metrics.throughput
      }].slice(-100)); // Keep last 100 points
    }
  }, [metrics]);

  const MetricCard = ({ title, value, icon: Icon, unit, color }) => (
    <div className="bg-white p-4 rounded-lg shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-xl font-semibold">{value} {unit}</p>
        </div>
        <Icon size={24} className={`text-${color}-500`} />
      </div>
    </div>
  );

  const renderChart = () => {
    const chartData = historicalData.map(d => ({
      timestamp: d.timestamp,
      value: d[selectedMetric]
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="colorMetric" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={tick => new Date(tick).toLocaleTimeString()}
          />
          <YAxis />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-white p-2 border rounded shadow">
                    <p>{new Date(payload[0].payload.timestamp).toLocaleTimeString()}</p>
                    <p className="text-[#8884d8]">
                      {payload[0].value.toFixed(2)}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#8884d8"
            fillOpacity={1}
            fill="url(#colorMetric)"
          />
        </AreaChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          title="CPU Usage"
          value={metrics?.cpu_percent.toFixed(1)}
          icon={Cpu}
          unit="%"
          color="blue"
        />
        <MetricCard
          title="Memory Usage"
          value={metrics?.memory_percent.toFixed(1)}
          icon={Database}
          unit="%"
          color="green"
        />
        <MetricCard
          title="GPU Memory"
          value={(metrics?.gpu_memory_used / 1e9).toFixed(2)}
          icon={Activity}
          unit="GB"
          color="purple"
        />
        <MetricCard
          title="Throughput"
          value={metrics?.throughput.toFixed(2)}
          icon={TrendingUp}
          unit="tx/s"
          color="orange"
        />
      </div>

      <div className="bg-white p-4 rounded-lg shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Performance History</h3>
          <select
            className="px-3 py-1 border rounded"
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            <option value="cpu">CPU Usage</option>
            <option value="memory">Memory Usage</option>
            <option value="gpu">GPU Memory</option>
            <option value="throughput">Throughput</option>
          </select>
        </div>
        {renderChart()}
      </div>

      <div className="bg-white p-4 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Model Performance</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500">Inference Time</p>
            <p className="text-lg">{metrics?.avg_inference_time?.toFixed(2)} ms</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Batch Size</p>
            <p className="text-lg">{metrics?.avg_batch_size}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Memory Efficiency</p>
            <p className="text-lg">{(metrics?.memory_efficiency * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">GPU Utilization</p>
            <p className="text-lg">{(metrics?.gpu_utilization * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonitoringDashboard;