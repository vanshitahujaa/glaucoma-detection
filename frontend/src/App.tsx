import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';

interface PredictionResult {
  prediction: string;
  probabilities: Record<string, number>;
  heatmap: string;
}

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-blue-600 text-white p-6 shadow-md">
        <h1 className="text-3xl font-bold">Glaucoma Detection System</h1>
        <p className="mt-2">Upload retinal fundus images for glaucoma classification</p>
      </header>

      <main className="container mx-auto p-6 max-w-5xl">
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
          <ImageUpload 
            setResult={setResult} 
            setLoading={setLoading} 
            setError={setError} 
          />
        </div>

        {loading && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-center items-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              <p className="ml-3">Processing image...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-6">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {result && !loading && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Results</h2>
            <ResultDisplay result={result} />
          </div>
        )}
      </main>

      <footer className="bg-gray-800 text-white p-4 mt-auto">
        <div className="container mx-auto text-center">
          <p>AI-Powered Glaucoma Detection System</p>
        </div>
      </footer>
    </div>
  );
}

export default App;