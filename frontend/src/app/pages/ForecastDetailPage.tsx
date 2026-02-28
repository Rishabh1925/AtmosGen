import { Link, useParams } from 'react-router';
import { Download, Share2, MapPin, Calendar, Clock, Target, TrendingUp, ArrowLeft } from 'lucide-react';
import { motion } from 'motion/react';
import { ImageWithFallback } from '../components/figma/ImageWithFallback';
import { Navigation } from '../components/Navigation';

export function ForecastDetailPage() {
  const { id } = useParams();

  // Mock forecast data based on ID
  const forecast = {
    id: id || '1',
    location: 'San Francisco, CA',
    date: 'Feb 26, 2026',
    time: '14:30 PST',
    accuracy: '94.2%',
    confidence: '96.8%',
    processingTime: '2.1s',
    temperature: '68°F',
    humidity: '65%',
    windSpeed: '12 mph',
    pressure: '1013 mb',
    visibility: '10 mi',
    imageUrl:
      'https://images.unsplash.com/photo-1760492867541-998db0881e82?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3ZWF0aGVyJTIwY2xvdWRzJTIwc2t5JTIwYmx1ZXxlbnwxfHx8fDE3NzIxNjk1ODd8MA&ixlib=rb-4.1.0&q=80&w=1080',
  };

  const metrics = [
    { label: 'Temperature', value: forecast.temperature, icon: TrendingUp },
    { label: 'Humidity', value: forecast.humidity, icon: Target },
    { label: 'Wind Speed', value: forecast.windSpeed, icon: TrendingUp },
    { label: 'Pressure', value: forecast.pressure, icon: Target },
  ];

  return (
    <div className="min-h-screen">
      <Navigation />

      <div className="pt-24 pb-12 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-12 flex items-center justify-between"
          >
            <div>
              <Link
                to="/forecast"
                className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors mb-4"
              >
                <ArrowLeft className="size-4" />
                <span>Back to Forecast</span>
              </Link>
              <h1 className="mb-2">Forecast Details</h1>
              <p className="text-xl text-gray-600 dark:text-gray-400">{forecast.location}</p>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors">
                <Share2 className="size-5 text-gray-600 dark:text-gray-400" />
              </button>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2">
                <Download className="size-4" />
                Download
              </button>
            </div>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Forecast Image */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 overflow-hidden"
              >
                <div className="aspect-video">
                  <ImageWithFallback
                    src={forecast.imageUrl}
                    alt={forecast.location}
                    className="w-full h-full object-cover"
                  />
                </div>
              </motion.div>

              {/* Metrics Grid */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="grid grid-cols-2 md:grid-cols-4 gap-4"
              >
                {metrics.map((metric, index) => (
                  <div
                    key={metric.label}
                    className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-4"
                  >
                    <metric.icon className="size-6 text-blue-600 dark:text-blue-400 mb-2" />
                    <p className="text-2xl mb-1 text-gray-900 dark:text-white">{metric.value}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</p>
                  </div>
                ))}
              </motion.div>

              {/* Additional Details */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
              >
                <h3 className="mb-4 text-gray-900 dark:text-white">Forecast Details</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-gray-600 dark:text-gray-400">Visibility</span>
                    <span className="text-gray-900 dark:text-white">{forecast.visibility}</span>
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-gray-600 dark:text-gray-400">Processing Time</span>
                    <span className="text-gray-900 dark:text-white">{forecast.processingTime}</span>
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-gray-600 dark:text-gray-400">Confidence Level</span>
                    <span className="text-green-600 dark:text-green-400">{forecast.confidence}</span>
                  </div>
                  <div className="flex items-center justify-between py-3">
                    <span className="text-gray-600 dark:text-gray-400">Model Accuracy</span>
                    <span className="text-blue-600 dark:text-blue-400">{forecast.accuracy}</span>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Location Info */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
              >
                <h3 className="mb-4 text-gray-900 dark:text-white">Location</h3>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <MapPin className="size-5 text-gray-400" />
                    <span className="text-gray-900 dark:text-white">{forecast.location}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Calendar className="size-5 text-gray-400" />
                    <span className="text-gray-900 dark:text-white">{forecast.date}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Clock className="size-5 text-gray-400" />
                    <span className="text-gray-900 dark:text-white">{forecast.time}</span>
                  </div>
                </div>
              </motion.div>

              {/* Accuracy Score */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
              >
                <h3 className="mb-4 text-gray-900 dark:text-white">Accuracy Score</h3>
                <div className="text-center">
                  <div className="relative inline-flex items-center justify-center w-32 h-32 mb-4">
                    <svg className="w-full h-full transform -rotate-90">
                      <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        className="text-gray-200 dark:text-gray-700"
                      />
                      <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        strokeDasharray={`${2 * Math.PI * 56}`}
                        strokeDashoffset={`${2 * Math.PI * 56 * (1 - 0.942)}`}
                        className="text-blue-600 dark:text-blue-400"
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-3xl text-gray-900 dark:text-white">{forecast.accuracy}</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Excellent prediction accuracy based on historical data validation
                  </p>
                </div>
              </motion.div>

              {/* Actions */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
              >
                <h3 className="mb-4 text-gray-900 dark:text-white">Actions</h3>
                <div className="space-y-3">
                  <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Save to Reports
                  </button>
                  <button className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                    Create Alert
                  </button>
                  <button className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                    Export Data
                  </button>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
