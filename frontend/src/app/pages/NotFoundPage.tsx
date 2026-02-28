import { Link } from 'react-router';
import { Cloud, ArrowLeft } from 'lucide-react';
import { motion } from 'motion/react';

export function NotFoundPage() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <Cloud className="size-20 text-blue-600 dark:text-blue-400 mx-auto mb-6 opacity-50" />
        <h1 className="text-6xl mb-4 text-gray-900 dark:text-white">404</h1>
        <h2 className="mb-4 text-gray-900 dark:text-white">Page not found</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <ArrowLeft className="size-4" />
          Back to Home
        </Link>
      </motion.div>
    </div>
  );
}