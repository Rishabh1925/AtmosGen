import React, { useState, useEffect } from 'react';
import { useAuth } from '../../lib/auth';
import { api } from '../../lib/api';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/ui/GlassCard';
import { Button } from '../components/ui/button';
import { Calendar, Clock, Satellite, Upload, Eye, Trash2, Search } from 'lucide-react';
import { motion } from 'motion/react';
import { Link } from 'react-router';

interface Forecast {
  id: number;
  name: string | null;
  input_images_count: number;
  processing_time: number;
  created_at: string;
}

export function ForecastHistoryPage() {
  const { user } = useAuth();
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'processing_time'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    if (user) {
      loadForecasts();
    }
  }, [user]);

  const loadForecasts = async () => {
    try {
      const response = await api.get('/forecasts');
      if (response.data.success) {
        setForecasts(response.data.forecasts);
      }
    } catch (error) {
      console.error('Failed to load forecasts:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredAndSortedForecasts = forecasts
    .filter(forecast => 
      !searchTerm || 
      forecast.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      new Date(forecast.created_at).toLocaleDateString().includes(searchTerm)
    )
    .sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'name':
          aValue = a.name || 'Untitled';
          bValue = b.name || 'Untitled';
          break;
        case 'processing_time':
          aValue = a.processing_time;
          bValue = b.processing_time;
          break;
        case 'date':
        default:
          aValue = new Date(a.created_at).getTime();
          bValue = new Date(b.created_at).getTime();
          break;
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

  const stats = {
    total: forecasts.length,
    thisWeek: forecasts.filter(f => {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      return new Date(f.created_at) > weekAgo;
    }).length,
    avgProcessingTime: forecasts.length > 0 
      ? forecasts.reduce((sum, f) => sum + f.processing_time, 0) / forecasts.length 
      : 0,
    satelliteForecasts: forecasts.filter(f => f.name?.includes('Satellite')).length
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
        <Navigation />
        <div className="pt-24 pb-12 px-6 flex items-center justify-center">
          <GlassCard className="p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Authentication Required</h2>
            <p className="text-gray-300 mb-6">Please log in to view your forecast history.</p>
            <Link to="/login">
              <Button className="bg-blue-600 hover:bg-blue-700">
                Login
              </Button>
            </Link>
          </GlassCard>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      <Navigation />
      
      <div className="pt-24 pb-12 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <h1 className="text-4xl font-bold text-white mb-4">Forecast History</h1>
            <p className="text-xl text-gray-300">
              View and manage all your weather forecasts
            </p>
          </motion.div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <GlassCard className="p-6 text-center">
                <div className="text-3xl font-bold text-white mb-2">{stats.total}</div>
                <div className="text-gray-300 text-sm">Total Forecasts</div>
              </GlassCard>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <GlassCard className="p-6 text-center">
                <div className="text-3xl font-bold text-white mb-2">{stats.thisWeek}</div>
                <div className="text-gray-300 text-sm">This Week</div>
              </GlassCard>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <GlassCard className="p-6 text-center">
                <div className="text-3xl font-bold text-white mb-2">
                  {stats.avgProcessingTime.toFixed(1)}s
                </div>
                <div className="text-gray-300 text-sm">Avg Processing Time</div>
              </GlassCard>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <GlassCard className="p-6 text-center">
                <div className="text-3xl font-bold text-white mb-2 flex items-center justify-center gap-2">
                  <Satellite className="w-8 h-8" />
                  {stats.satelliteForecasts}
                </div>
                <div className="text-gray-300 text-sm">Satellite Forecasts</div>
              </GlassCard>
            </motion.div>
          </div>

          {/* Controls */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <GlassCard className="p-6 mb-8">
              <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
                {/* Search */}
                <div className="relative flex-1 max-w-md">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <input
                    type="text"
                    placeholder="Search forecasts..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* Sort Controls */}
                <div className="flex gap-2">
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="date">Sort by Date</option>
                    <option value="name">Sort by Name</option>
                    <option value="processing_time">Sort by Processing Time</option>
                  </select>
                  
                  <button
                    onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                    className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white hover:bg-white/20 transition-colors"
                  >
                    {sortOrder === 'asc' ? '↑' : '↓'}
                  </button>
                </div>
              </div>
            </GlassCard>
          </motion.div>

          {/* Forecasts List */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <GlassCard className="p-6">
              {loading ? (
                <div className="text-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                  <p className="text-gray-300">Loading forecasts...</p>
                </div>
              ) : filteredAndSortedForecasts.length === 0 ? (
                <div className="text-center py-12">
                  <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">No Forecasts Yet</h3>
                  <p className="text-gray-300 mb-6">
                    {searchTerm ? 'No forecasts match your search.' : 'Start by creating your first weather forecast.'}
                  </p>
                  {!searchTerm && (
                    <div className="flex gap-4 justify-center">
                      <Link to="/forecast">
                        <Button className="bg-blue-600 hover:bg-blue-700">
                          Upload Images
                        </Button>
                      </Link>
                      <Link to="/satellite">
                        <Button className="bg-green-600 hover:bg-green-700">
                          <Satellite className="w-4 h-4 mr-2" />
                          Use Satellite Data
                        </Button>
                      </Link>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  {filteredAndSortedForecasts.map((forecast, index) => (
                    <motion.div
                      key={forecast.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-white/5 border border-white/10 rounded-lg p-4 hover:bg-white/10 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            {forecast.name?.includes('Satellite') ? (
                              <Satellite className="w-5 h-5 text-green-400" />
                            ) : (
                              <Upload className="w-5 h-5 text-blue-400" />
                            )}
                            <h3 className="text-lg font-semibold text-white">
                              {forecast.name || 'Untitled Forecast'}
                            </h3>
                          </div>
                          
                          <div className="flex items-center gap-6 text-sm text-gray-300">
                            <div className="flex items-center gap-1">
                              <Calendar className="w-4 h-4" />
                              {new Date(forecast.created_at).toLocaleDateString()}
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4" />
                              {forecast.processing_time.toFixed(1)}s
                            </div>
                            <div>
                              {forecast.input_images_count} images
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Link to={`/forecast/${forecast.id}`}>
                            <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                              <Eye className="w-4 h-4 mr-1" />
                              View
                            </Button>
                          </Link>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
}