import { Link } from 'react-router';
import { Upload, TrendingUp, Cloud, Zap } from 'lucide-react';
import { motion } from 'motion/react';
import { ImageWithFallback } from '../components/figma/ImageWithFallback';
import { Navigation } from '../components/Navigation';

export function HomePage() {
  const recentForecasts = [
    {
      id: '1',
      location: 'San Francisco, CA',
      date: 'Feb 26, 2026',
      accuracy: '94.2%',
      imageUrl: 'https://images.unsplash.com/photo-1760492867541-998db0881e82?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3ZWF0aGVyJTIwY2xvdWRzJTIwc2t5JTIwYmx1ZXxlbnwxfHx8fDE3NzIxNjk1ODd8MA&ixlib=rb-4.1.0&q=80&w=400',
    },
    {
      id: '2',
      location: 'New York, NY',
      date: 'Feb 25, 2026',
      accuracy: '91.8%',
      imageUrl: 'https://images.unsplash.com/photo-1579619114912-fda6dc30283f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3ZWF0aGVyJTIwZm9yZWNhc3QlMjBzdG9ybXxlbnwxfHx8fDE3NzIxNjk1ODh8MA&ixlib=rb-4.1.0&q=80&w=400',
    },
    {
      id: '3',
      location: 'Miami, FL',
      date: 'Feb 24, 2026',
      accuracy: '96.5%',
      imageUrl: 'https://images.unsplash.com/photo-1573490647695-2892d0bf89e7?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzYXRlbGxpdGUlMjB3ZWF0aGVyJTIwaW1hZ2VyeXxlbnwxfHx8fDE3NzIxNjk1ODh8MA&ixlib=rb-4.1.0&q=80&w=400',
    },
  ];

  return (
    <div className="min-h-screen">
      <Navigation />

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center max-w-3xl mx-auto mb-16"
          >
            <h1 className="text-5xl mb-6 tracking-tight">
              AI-Powered Weather Forecasting
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
              Enterprise-grade weather prediction using advanced machine learning and satellite imagery analysis
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                to="/register"
                className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Start Forecasting
              </Link>
              <Link
                to="/dashboard"
                className="px-8 py-3 bg-white/70 dark:bg-gray-800/70 backdrop-blur-sm border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-white dark:hover:bg-gray-800 transition-colors"
              >
                View Dashboard
              </Link>
            </div>
          </motion.div>

          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="max-w-4xl mx-auto"
          >
            <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-2xl border border-white/20 dark:border-gray-700/20 shadow-xl p-12">
              <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-12 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors cursor-pointer">
                <Upload className="size-12 text-gray-400 mx-auto mb-4" />
                <h3 className="mb-2 text-gray-900 dark:text-gray-100">Upload Satellite Imagery</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Drop your satellite images here or click to browse
                </p>
                <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  Select Files
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="grid md:grid-cols-3 gap-6 mb-20"
          >
            <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-8">
              <Zap className="size-10 text-blue-600 mb-4" />
              <h3 className="mb-3">Real-time Processing</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Get accurate forecasts in seconds with our advanced AI models
              </p>
            </div>
            <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-8">
              <TrendingUp className="size-10 text-blue-600 mb-4" />
              <h3 className="mb-3">95%+ Accuracy</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Industry-leading prediction accuracy powered by deep learning
              </p>
            </div>
            <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-8">
              <Cloud className="size-10 text-blue-600 mb-4" />
              <h3 className="mb-3">Global Coverage</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Support for weather data from anywhere in the world
              </p>
            </div>
          </motion.div>

          {/* Recent Forecasts */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-8 text-center">Recent Forecasts</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {recentForecasts.map((forecast, index) => (
                <motion.div
                  key={forecast.id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 overflow-hidden hover:shadow-lg transition-shadow"
                >
                  <div className="aspect-video overflow-hidden">
                    <ImageWithFallback
                      src={forecast.imageUrl}
                      alt={forecast.location}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="p-6">
                    <h4 className="mb-2">{forecast.location}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{forecast.date}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Accuracy</span>
                      <span className="text-sm text-blue-600">{forecast.accuracy}</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-2xl border border-white/20 dark:border-gray-700/20 p-12"
          >
            <h2 className="mb-4">Ready to get started?</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
              Join leading meteorology teams using AtmosGen
            </p>
            <Link
              to="/register"
              className="inline-block px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Create Account
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200/50 dark:border-gray-700/50 py-8 px-6">
        <div className="max-w-7xl mx-auto text-center text-gray-600 dark:text-gray-400">
          <p>&copy; 2026 AtmosGen. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
