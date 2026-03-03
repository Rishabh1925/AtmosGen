import { Navigation } from '../components/Navigation';
import { ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';

const faqs = [
    {
        question: 'What is AtmosGen?',
        answer:
            'AtmosGen is an AI-powered cloud coverage prediction platform. It uses a fine-tuned U-Net segmentation model with an EfficientNet-B0 encoder to analyze satellite imagery and predict cloud coverage percentage with pixel-level accuracy.',
    },
    {
        question: 'What kind of images can I upload for prediction?',
        answer:
            'You can upload satellite imagery in common formats such as PNG, JPG, or TIFF. The model works best with GOES-18 satellite band imagery, but it can also produce reasonable results on other satellite images. The image is automatically resized to the model\'s input dimensions during inference.',
    },
    {
        question: 'How accurate is the cloud coverage prediction?',
        answer:
            'The U-Net model with EfficientNet-B0 encoder was trained on labeled satellite data and achieves strong segmentation performance. Accuracy depends on the quality and type of input imagery. For GOES-18 band data, the model produces highly reliable cloud masks and coverage percentages.',
    },
    {
        question: 'Do I need to create an account to use AtmosGen?',
        answer:
            'You can access the forecast page without an account. However, creating a free account lets you save your prediction history, access the analytics dashboard, and track cloud coverage trends over time.',
    },
    {
        question: 'What does the dashboard show?',
        answer:
            'The dashboard provides an overview of your prediction activity including total forecasts run, average cloud coverage across predictions, average processing time, a bar chart of recent coverage results, and a gradient area chart showing your coverage trend over time.',
    },
    {
        question: 'How long does a prediction take?',
        answer:
            'Predictions typically complete in under 1 second. The processing time depends on your server hardware — with a GPU, inference is near-instantaneous. The exact processing time is displayed in the results for each prediction.',
    },
    {
        question: 'What model architecture does AtmosGen use?',
        answer:
            'AtmosGen uses a U-Net architecture with an EfficientNet-B0 encoder from the segmentation_models_pytorch library. This combination provides an excellent balance between accuracy and inference speed for binary cloud segmentation tasks.',
    },
    {
        question: 'Can I run AtmosGen locally?',
        answer:
            'Yes! AtmosGen is fully open source. Clone the repository from GitHub, install the Python backend dependencies, run the FastAPI server, and start the React frontend with npm. Full setup instructions are in the README.',
    },
    {
        question: 'What technologies power AtmosGen?',
        answer:
            'The backend is built with FastAPI and PyTorch, using MongoDB for data storage and JWT for authentication. The frontend uses React 19 with TypeScript, Tailwind CSS, Recharts for data visualization, Framer Motion for animations, and shadcn/ui components.',
    },
    {
        question: 'How can I contribute to AtmosGen?',
        answer:
            'AtmosGen is open source and contributions are welcome! Visit our GitHub repository at github.com/Rishabh1925/AtmosGen, fork the project, create a feature branch, and submit a pull request. You can also open issues for bug reports or feature requests.',
    },
];

export function FAQPage() {
    const [openIndex, setOpenIndex] = useState<number | null>(null);

    const toggle = (index: number) => {
        setOpenIndex(openIndex === index ? null : index);
    };

    return (
        <div className="min-h-screen">
            <Navigation />

            <div className="pt-24 pb-12 px-6">
                <div className="max-w-4xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                        className="text-center mb-16"
                    >
                        <h1 className="text-5xl md:text-6xl mb-6 text-gray-900 dark:text-white">
                            Frequently Asked Questions
                        </h1>
                        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
                            Everything you need to know about AtmosGen
                        </p>
                    </motion.div>

                    <div className="space-y-4">
                        {faqs.map((faq, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.4, delay: index * 0.05 }}
                            >
                                <div className="backdrop-blur-lg bg-white/60 dark:bg-gray-800/60 rounded-xl border border-white/20 dark:border-gray-700/20 overflow-hidden">
                                    <button
                                        onClick={() => toggle(index)}
                                        className="w-full flex items-center justify-between p-6 text-left"
                                    >
                                        <span className="text-lg font-medium text-gray-900 dark:text-white pr-4">
                                            {faq.question}
                                        </span>
                                        <ChevronDown
                                            className={`size-5 text-gray-500 dark:text-gray-400 shrink-0 transition-transform duration-200 ${openIndex === index ? 'rotate-180' : ''
                                                }`}
                                        />
                                    </button>
                                    <AnimatePresence>
                                        {openIndex === index && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: 'auto', opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                transition={{ duration: 0.2 }}
                                                className="overflow-hidden"
                                            >
                                                <div className="px-6 pb-6 text-gray-600 dark:text-gray-400 leading-relaxed">
                                                    {faq.answer}
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
