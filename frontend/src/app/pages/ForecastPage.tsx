import { Link } from 'react-router';
import { Upload, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';
import { Navigation } from '../components/Navigation';
import { useState, useRef } from 'react';
import { api } from '../../lib/api';

export function ForecastPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      const response = await api.generateForecast([selectedFile]);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to generate forecast. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen">
      <Navigation />

      <div className="pt-24 pb-12 px-6">
        <div className="max-w-5xl mx-auto">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl md:text-6xl mb-6 text-gray-900 dark:text-white">
              AI-Powered Weather Forecasting
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto mb-8">
              Enterprise-grade weather prediction using advanced machine learning and satellite imagery analysis
            </p>
            <div className="flex items-center justify-center gap-4">
              <button 
                onClick={() => document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' })}
                className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg"
              >
                Start Forecasting
              </button>
              <Link
                to="/dashboard"
                className="px-8 py-3 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-lg"
              >
                View Dashboard
              </Link>
            </div>
          </motion.div>

          {/* Upload Section */}
          <motion.div
            id="upload-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-2xl border border-white/20 dark:border-gray-700/20 p-12"
          >
            {!result ? (
              <div 
                className="border-2 border-dashed border-gray-400 dark:border-gray-600 rounded-xl p-16 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors cursor-pointer"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="size-16 text-gray-400 dark:text-gray-500 mx-auto mb-6" strokeWidth={1.5} />
                <h3 className="text-2xl mb-3 text-gray-900 dark:text-white">Upload Satellite Imagery</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Drop your satellite images here or click to browse
                </p>
                
                {selectedFile ? (
                  <div className="mb-6">
                    <div className="flex items-center justify-center gap-2 mb-4">
                      <CheckCircle className="size-5 text-green-500" />
                      <span className="text-gray-700 dark:text-gray-300">{selectedFile.name}</span>
                    </div>
                    <div className="flex gap-4 justify-center">
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleUpload();
                        }}
                        disabled={isUploading}
                        className="px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-lg disabled:opacity-50"
                      >
                        {isUploading ? 'Generating...' : 'Generate Forecast'}
                      </button>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          resetUpload();
                        }}
                        className="px-8 py-3 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-lg"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                ) : (
                  <button className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg">
                    Select Files
                  </button>
                )}

                {error && (
                  <div className="mt-4 flex items-center justify-center gap-2 text-red-500">
                    <AlertCircle className="size-5" />
                    <span>{error}</span>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileInputChange}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="text-center">
                <h3 className="text-2xl mb-6 text-gray-900 dark:text-white">Weather Forecast Generated</h3>
                
                <div className="mb-6">
                  <img 
                    src={`data:image/png;base64,${result.generated_image}`}
                    alt="Generated weather forecast"
                    className="max-w-full h-auto rounded-lg mx-auto shadow-lg"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 text-sm">
                  <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                    <div className="font-semibold text-gray-900 dark:text-white">Processing Time</div>
                    <div className="text-gray-600 dark:text-gray-400">{result.processing_time.toFixed(3)}s</div>
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                    <div className="font-semibold text-gray-900 dark:text-white">Model Type</div>
                    <div className="text-gray-600 dark:text-gray-400">Lightweight Weather CNN</div>
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                    <div className="font-semibold text-gray-900 dark:text-white">Status</div>
                    <div className="text-green-600 dark:text-green-400">Success</div>
                  </div>
                </div>

                <div className="flex gap-4 justify-center">
                  <button 
                    onClick={resetUpload}
                    className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg"
                  >
                    Generate Another
                  </button>
                  <Link
                    to="/dashboard"
                    className="px-8 py-3 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-lg"
                  >
                    View Dashboard
                  </Link>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
