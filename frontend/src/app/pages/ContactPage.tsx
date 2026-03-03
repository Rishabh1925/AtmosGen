import { Navigation } from '../components/Navigation';
import { Mail, Send, Github, MessageCircleQuestion } from 'lucide-react';
import { Link } from 'react-router';
import { motion } from 'motion/react';
import { useState } from 'react';

export function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const subject = encodeURIComponent(`[AtmosGen] ${formData.subject}`);
    const body = encodeURIComponent(
      `Hi AtmosGen Team,\n\n` +
      `${formData.message}\n\n` +
      `---\n` +
      `Name: ${formData.name}\n` +
      `Email: ${formData.email}\n` +
      `Subject: ${formData.subject}\n` +
      `Sent from AtmosGen Contact Form`
    );
    window.location.href = `mailto:atmosgenhelp@gmail.com?subject=${subject}&body=${body}`;
  };

  const updateField = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

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
            <h1 className="text-5xl md:text-6xl mb-6 text-gray-900 dark:text-white">Contact Us</h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Have questions about AtmosGen? We're here to help
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Contact Form */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="lg:col-span-2"
            >
              <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-8">
                <h3 className="mb-6 text-gray-900 dark:text-white text-lg font-semibold">Send us a message</h3>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="name" className="block mb-2 text-gray-700 dark:text-gray-300">
                        Name
                      </label>
                      <input
                        id="name"
                        type="text"
                        value={formData.name}
                        onChange={(e) => updateField('name', e.target.value)}
                        placeholder="John Doe"
                        className="w-full px-4 py-3 bg-white/50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="email" className="block mb-2 text-gray-700 dark:text-gray-300">
                        Email
                      </label>
                      <input
                        id="email"
                        type="email"
                        value={formData.email}
                        onChange={(e) => updateField('email', e.target.value)}
                        placeholder="you@example.com"
                        className="w-full px-4 py-3 bg-white/50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                        required
                      />
                    </div>
                  </div>

                  <div>
                    <label htmlFor="subject" className="block mb-2 text-gray-700 dark:text-gray-300">
                      Subject
                    </label>
                    <input
                      id="subject"
                      type="text"
                      value={formData.subject}
                      onChange={(e) => updateField('subject', e.target.value)}
                      placeholder="How can we help?"
                      className="w-full px-4 py-3 bg-white/50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                      required
                    />
                  </div>

                  <div>
                    <label htmlFor="message" className="block mb-2 text-gray-700 dark:text-gray-300">
                      Message
                    </label>
                    <textarea
                      id="message"
                      value={formData.message}
                      onChange={(e) => updateField('message', e.target.value)}
                      placeholder="Tell us more about your inquiry..."
                      rows={6}
                      className="w-full px-4 py-3 bg-white/50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
                      required
                    />
                  </div>

                  <button
                    type="submit"
                    className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                  >
                    <Send className="size-4" />
                    Send Message
                  </button>
                </form>
              </div>
            </motion.div>

            {/* Sidebar — Email, GitHub, FAQ */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="space-y-6"
            >
              {/* Email */}
              <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-blue-100 dark:bg-blue-900/40 rounded-lg">
                    <Mail className="size-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-1">Email</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Send us an email anytime</p>
                    <a
                      href="mailto:atmosgenhelp@gmail.com"
                      className="text-blue-600 dark:text-blue-400 hover:underline text-sm font-medium"
                    >
                      atmosgenhelp@gmail.com
                    </a>
                  </div>
                </div>
              </div>

              {/* GitHub */}
              <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-blue-100 dark:bg-blue-900/40 rounded-lg">
                    <Github className="size-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-1">GitHub</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Check out our code and contribute</p>
                    <a
                      href="https://github.com/Rishabh1925/AtmosGen"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 dark:text-blue-400 hover:underline text-sm font-medium"
                    >
                      View Repository
                    </a>
                  </div>
                </div>
              </div>

              {/* FAQ */}
              <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-blue-100 dark:bg-blue-900/40 rounded-lg">
                    <MessageCircleQuestion className="size-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-1">FAQ</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Find answers to common questions</p>
                    <Link
                      to="/faq"
                      className="text-blue-600 dark:text-blue-400 hover:underline text-sm font-medium inline-flex items-center gap-1"
                    >
                      View All FAQs
                    </Link>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
