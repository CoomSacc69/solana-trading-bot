import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { Tab } from '@headlessui/react';
import MonitoringDashboard from './components/MonitoringDashboard';
import metricsHandler from './background/metrics';

const Popup = () => {
  const [metrics, setMetrics] = useState(null);
  
  useEffect(() => {
    const unsubscribe = metricsHandler.subscribe(setMetrics);
    metricsHandler.connect();
    return unsubscribe;
  }, []);

  return (
    <div className="w-[800px] h-[600px] p-4 bg-gray-50">
      <Tab.Group>
        <Tab.List className="flex space-x-1 rounded-xl bg-blue-900/20 p-1 mb-4">
          <Tab 
            className={({ selected }) =>
              `w-full rounded-lg py-2.5 text-sm font-medium leading-5 
               ${selected 
                 ? 'bg-white text-blue-700 shadow'
                 : 'text-blue-100 hover:bg-white/[0.12] hover:text-white'
               }`
            }
          >
            Dashboard
          </Tab>
          <Tab 
            className={({ selected }) =>
              `w-full rounded-lg py-2.5 text-sm font-medium leading-5 
               ${selected 
                 ? 'bg-white text-blue-700 shadow'
                 : 'text-blue-100 hover:bg-white/[0.12] hover:text-white'
               }`
            }
          >
            Settings
          </Tab>
        </Tab.List>
        <Tab.Panels>
          <Tab.Panel>
            <MonitoringDashboard metrics={metrics} />
          </Tab.Panel>
          <Tab.Panel>
            {/* Settings panel content */}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
};

ReactDOM.render(
  <React.StrictMode>
    <Popup />
  </React.StrictMode>,
  document.getElementById('app')
);