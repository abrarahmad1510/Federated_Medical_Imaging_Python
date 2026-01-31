import React from 'react';
import { Routes as RouterRoutes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Experiments from './pages/Experiments';
import Models from './pages/Models';
import Clients from './pages/Clients';
import Monitoring from './pages/Monitoring';
import Login from './pages/Login';

const Routes: React.FC = () => {
  return (
    <RouterRoutes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/experiments" element={<Experiments />} />
      <Route path="/models" element={<Models />} />
      <Route path="/clients" element={<Clients />} />
      <Route path="/monitoring" element={<Monitoring />} />
      <Route path="/login" element={<Login />} />
      <Route path="*" element={<Navigate to="/" />} />
    </RouterRoutes>
  );
};

export default Routes;
