import React from 'react';

interface PredictionResult {
  prediction: string;
  probabilities: Record<string, number>;
  heatmap: string;
}

interface ResultDisplayProps {
  result: PredictionResult;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  const { prediction, probabilities, heatmap } = result;
  
  // Determine color based on prediction
  const getPredictionColor = (pred: string): string => {
    switch (pred.toLowerCase()) {
      case 'normal':
        return 'text-green-600';
      case 'suspicious':
        return 'text-yellow-600';
      case 'early':
        return 'text-orange-600';
      case 'advanced':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h3 className="text-lg font-medium mb-3">Prediction</h3>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-lg">
            Classification: <span className={`font-bold ${getPredictionColor(prediction)}`}>{prediction}</span>
          </p>
          
          <div className="mt-4">
            <h4 className="text-md font-medium mb-2">Probability Distribution</h4>
            <div className="space-y-2">
              {Object.entries(probabilities).map(([className, probability]) => (
                <div key={className} className="flex items-center">
                  <span className="w-24 text-sm">{className}:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-4">
                    <div 
                      className={`h-4 rounded-full ${getPredictionColor(className)}`}
                      style={{ width: `${probability * 100}%` }}
                    ></div>
                  </div>
                  <span className="ml-2 text-sm">{(probability * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-medium mb-3">Grad-CAM Heatmap</h3>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-sm mb-2">
            Regions highlighted in red indicate areas that influenced the model's prediction the most.
          </p>
          {heatmap && (
            <img 
              src={`data:image/jpeg;base64,${heatmap}`} 
              alt="Grad-CAM Heatmap" 
              className="w-full rounded-lg border border-gray-300"
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;