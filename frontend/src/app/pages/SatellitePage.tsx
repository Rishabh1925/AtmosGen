import React, { useState, useEffect } from 'react';
import { useAuth } from '../../lib/auth';
import { api } from '../../lib/api';
import { GlassCard } from '../components/ui/GlassCard';
import { Button } from '../components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Loader2, Satellite, MapPin, Calendar, Layers } from 'lucide-react';
import { toast } from 'sonner';

interface SatelliteImage {
  date: string;
  layer: string;
  region: string;
  image_data: string;
  source: string;
}

interface Region {
  bbox: number[];
  name: string;
}

export function SatellitePage() {
  const { user } = useAuth();
  const [regions, setRegions] = useState<Record<string, Region>>({});
  const [layers, setLayers] = useState<Record<string, string>>({});
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  
  const [selectedRegion, setSelectedRegion] = useState<string>('');
  const [selectedLayer, setSelectedLayer] = useState<string>('visible');
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [sequenceLength, setSequenceLength] = useState<number>(4);
  const [forecastName, setForecastName] = useState<string>('');
  
  const [satelliteImages, setSatelliteImages] = useState<SatelliteImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedForecast, setGeneratedForecast] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [regionsRes, layersRes, datesRes] = await Promise.all([
        api.get('/satellite/regions'),
        api.get('/satellite/layers'),
        api.get('/satellite/dates?days_back=14')
      ]);

      setRegions(regionsRes.data.regions);
      setLayers(layersRes.data.layers);
      setAvailableDates(datesRes.data.dates);
      
      // Set defaults
      if (regionsRes.data.regions) {
        setSelectedRegion(Object.keys(regionsRes.data.regions)[0]);
      }
      if (datesRes.data.dates.length > 0) {
        setSelectedDate(datesRes.data.dates[0]);
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
      toast.error('Failed to load satellite data options');
    }
  };

  const fetchSatelliteImages = async () => {
    if (!selectedRegion || !selectedDate) {
      toast.error('Please select region and date');
      return;
    }

    setIsLoading(true);
    try {
      const response = await api.get('/satellite/sequence', {
        params: {
          region: selectedRegion,
          layer: selectedLayer,
          sequence_length: sequenceLength,
          end_date: selectedDate
        }
      });

      setSatelliteImages(response.data.images);
      toast.success(`Loaded ${response.data.images.length} satellite images`);
    } catch (error) {
      console.error('Failed to fetch satellite images:', error);
      toast.error('Failed to fetch satellite images');
    } finally {
      setIsLoading(false);
    }
  };

  const generateForecast = async () => {
    if (satelliteImages.length === 0) {
      toast.error('Please fetch satellite images first');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await api.post('/predict/satellite', {
        region: selectedRegion,
        layer: selectedLayer,
        sequence_length: sequenceLength,
        end_date: selectedDate,
        forecast_name: forecastName || `Satellite Forecast ${new Date().toLocaleDateString()}`
      });

      setGeneratedForecast(response.data.generated_image);
      toast.success('Weather forecast generated successfully!');
    } catch (error) {
      console.error('Failed to generate forecast:', error);
      toast.error('Failed to generate forecast');
    } finally {
      setIsGenerating(false);
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 flex items-center justify-center">
        <GlassCard className="p-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Authentication Required</h2>
          <p className="text-gray-300">Please log in to access satellite data.</p>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center gap-3">
            <Satellite className="w-10 h-10" />
            Real Satellite Data
          </h1>
          <p className="text-xl text-gray-300">
            Generate weather forecasts using real-time satellite imagery from NASA and NOAA
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <GlassCard className="p-6">
              <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                <MapPin className="w-5 h-5" />
                Satellite Parameters
              </h2>

              <div className="space-y-4">
                {/* Region Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Region
                  </label>
                  <Select value={selectedRegion} onValueChange={setSelectedRegion}>
                    <SelectTrigger className="bg-white/10 border-white/20 text-white">
                      <SelectValue placeholder="Select region" />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(regions).map(([key, region]) => (
                        <SelectItem key={key} value={key}>
                          {region.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Layer Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                    <Layers className="w-4 h-4" />
                    Satellite Layer
                  </label>
                  <Select value={selectedLayer} onValueChange={setSelectedLayer}>
                    <SelectTrigger className="bg-white/10 border-white/20 text-white">
                      <SelectValue placeholder="Select layer" />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(layers).map(([key, name]) => (
                        <SelectItem key={key} value={key}>
                          {name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Date Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                    <Calendar className="w-4 h-4" />
                    End Date
                  </label>
                  <Select value={selectedDate} onValueChange={setSelectedDate}>
                    <SelectTrigger className="bg-white/10 border-white/20 text-white">
                      <SelectValue placeholder="Select date" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableDates.map((date) => (
                        <SelectItem key={date} value={date}>
                          {new Date(date).toLocaleDateString()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Sequence Length */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Sequence Length: {sequenceLength} days
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="7"
                    value={sequenceLength}
                    onChange={(e) => setSequenceLength(parseInt(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Forecast Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Forecast Name (Optional)
                  </label>
                  <input
                    type="text"
                    value={forecastName}
                    onChange={(e) => setForecastName(e.target.value)}
                    placeholder="My Satellite Forecast"
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* Action Buttons */}
                <div className="space-y-3 pt-4">
                  <Button
                    onClick={fetchSatelliteImages}
                    disabled={isLoading || !selectedRegion || !selectedDate}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Loading Images...
                      </>
                    ) : (
                      'Fetch Satellite Images'
                    )}
                  </Button>

                  <Button
                    onClick={generateForecast}
                    disabled={isGenerating || satelliteImages.length === 0}
                    className="w-full bg-green-600 hover:bg-green-700"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      'Generate Forecast'
                    )}
                  </Button>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Images Display */}
          <div className="lg:col-span-2">
            <GlassCard className="p-6">
              <h2 className="text-xl font-semibold text-white mb-6">
                Satellite Image Sequence
              </h2>

              {satelliteImages.length > 0 ? (
                <div className="grid grid-cols-2 gap-4 mb-6">
                  {satelliteImages.map((image, index) => (
                    <div key={index} className="relative">
                      <img
                        src={image.image_data}
                        alt={`Satellite ${image.date}`}
                        className="w-full h-48 object-cover rounded-lg border border-white/20"
                      />
                      <div className="absolute bottom-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-sm">
                        {new Date(image.date).toLocaleDateString()}
                      </div>
                      <div className="absolute top-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                        {image.source}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  <Satellite className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>No satellite images loaded yet.</p>
                  <p className="text-sm">Select parameters and click "Fetch Satellite Images"</p>
                </div>
              )}

              {/* Generated Forecast */}
              {generatedForecast && (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold text-white mb-4">Generated Forecast</h3>
                  <div className="relative">
                    <img
                      src={generatedForecast}
                      alt="Generated Weather Forecast"
                      className="w-full max-w-md mx-auto rounded-lg border border-white/20"
                    />
                    <div className="absolute top-2 left-2 bg-green-600 text-white px-3 py-1 rounded text-sm font-medium">
                      AI Forecast
                    </div>
                  </div>
                </div>
              )}
            </GlassCard>
          </div>
        </div>
      </div>
    </div>
  );
}