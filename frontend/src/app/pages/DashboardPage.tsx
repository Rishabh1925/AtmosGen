import { Navigation } from '../components/Navigation';
import { CloudRain, BarChart3, Clock } from 'lucide-react';
import { motion } from 'motion/react';
import {
  BarChart,
  Bar,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { useState, useEffect } from 'react';
import { api } from '../../lib/api';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '../components/ui/chart';

interface DashboardStats {
  total_forecasts: number;
  avg_cloud_coverage: number;
  avg_processing_time: number;
  recent_forecasts: Array<{
    id: string;
    cloud_coverage_pct: number;
    model_type: string;
    date: string;
  }>;
}

const coverageChartConfig = {
  coverage: {
    label: 'Cloud Coverage %',
    color: '#3b82f6',
  },
} satisfies ChartConfig;

const trendChartConfig = {
  coverage: {
    label: 'Coverage',
    color: '#3b82f6',
  },
  processingTime: {
    label: 'Processing (s)',
    color: '#8b5cf6',
  },
} satisfies ChartConfig;

export function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await api.get<DashboardStats>('/dashboard');
        setStats(response.data);
      } catch (err) {
        console.error('Failed to fetch dashboard stats:', err);
        setStats({
          total_forecasts: 0,
          avg_cloud_coverage: 0,
          avg_processing_time: 0,
          recent_forecasts: [],
        });
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  // Build chart data from recent forecasts
  const barData = stats?.recent_forecasts
    ?.slice()
    .reverse()
    .map((f, i) => ({
      label: `#${i + 1}`,
      coverage: f.cloud_coverage_pct,
    })) || [];

  // Build area chart data — coverage trend over time
  const areaData = stats?.recent_forecasts
    ?.slice()
    .reverse()
    .map((f) => ({
      date: new Date(f.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      coverage: f.cloud_coverage_pct,
    })) || [];

  const statCards = [
    {
      label: 'Avg Cloud Coverage',
      value: stats ? `${stats.avg_cloud_coverage}%` : '—',
      icon: CloudRain,
      color: 'text-blue-600',
    },
    {
      label: 'Total Forecasts',
      value: stats ? `${stats.total_forecasts}` : '—',
      icon: BarChart3,
      color: 'text-green-600',
    },
    {
      label: 'Avg Processing Time',
      value: stats ? `${stats.avg_processing_time}s` : '—',
      icon: Clock,
      color: 'text-purple-600',
    },
  ];

  return (
    <div className="min-h-screen">
      <Navigation />

      <div className="pt-24 pb-12 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl md:text-6xl mb-6 text-gray-900 dark:text-white">Dashboard</h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Cloud coverage prediction metrics and forecast history
            </p>
          </motion.div>

          {/* Stats Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            {statCards.map((stat, index) => (
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

          {/* Charts */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Coverage Bar Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
            >
              <h3 className="mb-2 text-gray-900 dark:text-white">Cloud Coverage — Recent Forecasts</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">Coverage percentage per analysis</p>
              {barData.length > 0 ? (
                <ChartContainer config={coverageChartConfig} className="h-[300px] w-full">
                  <BarChart data={barData}>
                    <CartesianGrid vertical={false} strokeDasharray="3 3" opacity={0.1} />
                    <XAxis dataKey="label" tickLine={false} axisLine={false} tickMargin={8} />
                    <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tickMargin={8} />
                    <ChartTooltip
                      cursor={false}
                      content={<ChartTooltipContent />}
                    />
                    <Bar dataKey="coverage" fill="var(--color-coverage)" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ChartContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-gray-500 dark:text-gray-400">
                  No forecast data yet — run a prediction first
                </div>
              )}
            </motion.div>

            {/* Coverage Trend Area Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6"
            >
              <h3 className="mb-2 text-gray-900 dark:text-white">Coverage Trend</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">Cloud coverage over time</p>
              {areaData.length > 0 ? (
                <>
                  <ChartContainer config={trendChartConfig} className="h-[260px] w-full">
                    <AreaChart
                      data={areaData}
                      margin={{ left: 12, right: 12 }}
                    >
                      <CartesianGrid vertical={false} strokeDasharray="3 3" opacity={0.1} />
                      <XAxis
                        dataKey="date"
                        tickLine={false}
                        axisLine={false}
                        tickMargin={8}
                      />
                      <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tickMargin={8} />
                      <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                      <defs>
                        <linearGradient id="fillCoverage" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--color-coverage)" stopOpacity={0.8} />
                          <stop offset="95%" stopColor="var(--color-coverage)" stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <Area
                        dataKey="coverage"
                        type="natural"
                        fill="url(#fillCoverage)"
                        fillOpacity={0.4}
                        stroke="var(--color-coverage)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ChartContainer>

                </>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-gray-500 dark:text-gray-400">
                  No forecast history yet
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
