import { simulate } from './engine.js';
self.onmessage = ({ data }) => {
  try { self.postMessage({ id: data.id, result: simulate(data.config) }); }
  catch (error) { self.postMessage({ id: data.id, error: error.message || 'The simulation could not be computed.' }); }
};
