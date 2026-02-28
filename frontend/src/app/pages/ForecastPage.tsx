import { Link } from 'react-router';
import { Upload } from 'lucide-react';
import { motion } from 'motion/react';
import { Navigation } from '../components/Navigation';

export function ForecastPage() {
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
              <button className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg">
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
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-2xl border border-white/20 dark:border-gray-700/20 p-12"
          >
            <div className="border-2 border-dashed border-gray-400 dark:border-gray-600 rounded-xl p-16 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors cursor-pointer">
              <Upload className="size-16 text-gray-400 dark:text-gray-500 mx-auto mb-6" strokeWidth={1.5} />
              <h3 className="text-2xl mb-3 text-gray-900 dark:text-white">Upload Satellite Imagery</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Drop your satellite images here or click to browse
              </p>
              <button className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg">
                Select Files
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
