import { useState, useEffect } from 'react';
import { api } from '../../lib/api';
import { Navigation } from '../components/Navigation';
import { Calendar, Clock, CloudRain, Upload, Eye, Search, ArrowUpDown } from 'lucide-react';
import { motion } from 'motion/react';
import { Link } from 'react-router';

interface Forecast {
  id: string;
  location?: string;
  cloud_coverage_pct?: number;
  model_type?: string;
  created_at: string;
  status?: string;
}

export function ForecastHistoryPage() {
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    loadForecasts();
  }, []);

  const loadForecasts = async () => {
    try {
      const response = await api.get<{ forecasts: Forecast[] }>('/forecasts');
      setForecasts(response.data.forecasts || []);
    } catch (error) {
      console.error('Failed to load forecasts:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredForecasts = forecasts
    .filter(f =>
      !searchTerm ||
      f.location?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.model_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      new Date(f.created_at).toLocaleDateString().includes(searchTerm)
    )
    .sort((a, b) => {
      const aTime = new Date(a.created_at).getTime();
      const bTime = new Date(b.created_at).getTime();
      return sortOrder === 'desc' ? bTime - aTime : aTime - bTime;
    });

  // Compute stats from data
  const stats = {
    total: forecasts.length,
    avgCoverage: forecasts.length > 0
      ? forecasts.reduce((sum, f) => sum + (f.cloud_coverage_pct || 0), 0) / forecasts.length
      : 0,
    thisWeek: forecasts.filter(f => {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      return new Date(f.created_at) > weekAgo;
    }).length,
  };

  return (
    <div className="min-h-screen">
      <Navigation />

      <div className="pt-24 pb-12 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl md:text-6xl mb-6 text-gray-900 dark:text-white">
              Forecast History
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              View all your cloud coverage predictions
            </p>
          </motion.div>

          {/* Stats Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {[
              { label: 'Total Forecasts', value: stats.total, icon: CloudRain, color: 'text-blue-600' },
              { label: 'Avg Coverage', value: `${stats.avgCoverage.toFixed(1)}%`, icon: CloudRain, color: 'text-green-600' },
              { label: 'This Week', value: stats.thisWeek, icon: Calendar, color: 'text-purple-600' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
              >
                <stat.icon className={`size-8 ${stat.color} mb-4`} />
                <p className="text-3xl mb-1 text-gray-900 dark:text-white">{loading ? '...' : stat.value}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>

          {/* Search & Sort Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-4 mb-6"
          >
            <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 size-4" />
                <input
                  type="text"
                  placeholder="Search forecasts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
              >
                <ArrowUpDown className="size-4" />
                {sortOrder === 'desc' ? 'Newest First' : 'Oldest First'}
              </button>
            </div>
          </motion.div>

          {/* Forecasts List */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
          >
            {loading ? (
              <div className="text-center py-16">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-500 dark:text-gray-400">Loading forecasts...</p>
              </div>
            ) : filteredForecasts.length === 0 ? (
              <div className="text-center py-16">
                <Upload className="size-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl mb-2 text-gray-900 dark:text-white">
                  {searchTerm ? 'No Matches' : 'No Forecasts Yet'}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  {searchTerm ? 'Try a different search term.' : 'Run your first cloud coverage prediction to see it here.'}
                </p>
                {!searchTerm && (
                  <Link
                    to="/forecast"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Start Forecasting
                  </Link>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredForecasts.map((forecast, index) => (
                  <motion.div
                    key={forecast.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <CloudRain className="size-5 text-blue-500" />
                        <span className="font-medium text-gray-900 dark:text-white">
                          {forecast.location || 'Cloud Analysis'}
                        </span>
                        {forecast.cloud_coverage_pct !== undefined && (
                          <span className="text-sm px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400">
                            {forecast.cloud_coverage_pct}% cloudy
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                        <span className="flex items-center gap-1">
                          <Calendar className="size-3.5" />
                          {new Date(forecast.created_at).toLocaleDateString()}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="size-3.5" />
                          {new Date(forecast.created_at).toLocaleTimeString()}
                        </span>
                        {forecast.model_type && (
                          <span className="text-xs text-gray-400 dark:text-gray-500">
                            {forecast.model_type}
                          </span>
                        )}
                      </div>
                    </div>
                    <Link
                      to={`/forecast/${forecast.id}`}
                      className="flex items-center gap-1 px-4 py-2 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors text-sm"
                    >
                      <Eye className="size-4" />
                      View
                    </Link>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}