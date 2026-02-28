import { Navigation } from '../components/Navigation';
import { BarChart3, TrendingUp, Activity, PieChart } from 'lucide-react';
import { motion } from 'motion/react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useTheme } from '../components/ThemeProvider';

export function DashboardPage() {
  const accuracyData = [
    { month: 'Jan', accuracy: 92.5, forecasts: 45 },
    { month: 'Feb', accuracy: 94.2, forecasts: 52 },
    { month: 'Mar', accuracy: 91.8, forecasts: 48 },
    { month: 'Apr', accuracy: 95.1, forecasts: 61 },
    { month: 'May', accuracy: 93.7, forecasts: 55 },
    { month: 'Jun', accuracy: 96.2, forecasts: 58 },
  ];

  const regionData = [
    { region: 'West Coast', accuracy: 94.5 },
    { region: 'East Coast', accuracy: 92.8 },
    { region: 'Midwest', accuracy: 93.2 },
    { region: 'South', accuracy: 95.1 },
  ];

  const { theme } = useTheme();

  return (
    <div className="min-h-screen">
      <Navigation />

      <div className="pt-24 pb-12 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-12"
          >
            <h1 className="mb-4">Dashboard</h1>
            <p className="text-xl text-gray-600 dark:text-gray-400">
              Comprehensive insights into forecast accuracy and performance metrics
            </p>
          </motion.div>

          {/* Stats Grid */}
          <div className="grid md:grid-cols-4 gap-6 mb-12">
            {[
              { label: 'Avg Accuracy', value: '94.2%', icon: TrendingUp, color: 'text-blue-600' },
              { label: 'Total Forecasts', value: '319', icon: BarChart3, color: 'text-green-600' },
              { label: 'Success Rate', value: '96.8%', icon: Activity, color: 'text-purple-600' },
              { label: 'Active Models', value: '12', icon: PieChart, color: 'text-orange-600' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
              >
                <stat.icon className={`size-8 ${stat.color} mb-4`} />
                <p className="text-3xl mb-1 text-gray-900 dark:text-white">{stat.value}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>

          {/* Charts */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Accuracy Trend */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
            >
              <h3 className="mb-6 text-gray-900 dark:text-white">Accuracy Trend</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="month" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                  <YAxis domain={[85, 100]} stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: theme === 'dark' ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                      border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                      borderRadius: '8px',
                      color: theme === 'dark' ? '#f3f4f6' : '#111827',
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={{ fill: '#2563eb', r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Regional Performance */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
            >
              <h3 className="mb-6 text-gray-900 dark:text-white">Regional Performance</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={regionData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="region" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                  <YAxis domain={[85, 100]} stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: theme === 'dark' ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                      border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                      borderRadius: '8px',
                      color: theme === 'dark' ? '#f3f4f6' : '#111827',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#2563eb" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
