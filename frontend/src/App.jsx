import React, { useState } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [selectedFiles, setSelectedFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)

  // Check backend health on component mount
  React.useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      setHealth(response.data)
    } catch (err) {
      setHealth({ status: 'error', model_loaded: false })
    }
  }

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files)
    
    if (files.length > 10) {
      setError('Maximum 10 images allowed')
      return
    }

    setSelectedFiles(files)
    setError(null)
    setResult(null)

    // Create previews
    const newPreviews = []
    files.forEach((file, index) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        newPreviews[index] = e.target.result
        if (newPreviews.length === files.length) {
          setPreviews([...newPreviews])
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const handlePredict = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      selectedFiles.forEach((file) => {
        formData.append('files', file)
      })

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 second timeout
      })

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const clearAll = () => {
    setSelectedFiles([])
    setPreviews([])
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🌤️ AtmosGen</h1>
        <p>Physics-Informed Diffusion Model for Satellite Weather Nowcasting</p>
        
        <div className="health-status">
          {health ? (
            <span className={`status ${health.status === 'healthy' && health.model_loaded ? 'healthy' : 'error'}`}>
              {health.status === 'healthy' && health.model_loaded ? '✅ Ready' : '❌ Not Ready'}
            </span>
          ) : (
            <span className="status loading">🔄 Checking...</span>
          )}
        </div>
      </header>

      <main className="main">
        <div className="upload-section">
          <h2>Upload Satellite Image Sequence</h2>
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-input"
              multiple
              accept="image/*"
              onChange={handleFileSelect}
              className="file-input"
            />
            <label htmlFor="file-input" className="file-input-label">
              📁 Select Images (Max 10)
            </label>
          </div>
          
          {selectedFiles.length > 0 && (
            <p className="file-count">{selectedFiles.length} image(s) selected</p>
          )}
        </div>

        {previews.length > 0 && (
          <div className="preview-section">
            <h3>Image Sequence Preview</h3>
            <div className="preview-grid">
              {previews.map((preview, index) => (
                <div key={index} className="preview-item">
                  <img src={preview} alt={`Preview ${index + 1}`} />
                  <span className="preview-label">Frame {index + 1}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="action-section">
          <button
            onClick={handlePredict}
            disabled={selectedFiles.length === 0 || isLoading || !health?.model_loaded}
            className="predict-button"
          >
            {isLoading ? '🔄 Generating Forecast...' : '🚀 Generate Forecast'}
          </button>
          
          {selectedFiles.length > 0 && (
            <button onClick={clearAll} className="clear-button">
              🗑️ Clear All
            </button>
          )}
        </div>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {result && (
          <div className="result-section">
            <h3>Generated Weather Forecast</h3>
            <div className="result-content">
              <img
                src={`data:image/png;base64,${result.generated_image}`}
                alt="Generated forecast"
                className="result-image"
              />
              <div className="result-info">
                <p>✅ {result.message}</p>
                <p>⏱️ Processing time: {result.processing_time.toFixed(2)}s</p>
                <p>📊 Input sequence: {selectedFiles.length} frames</p>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>AtmosGen v1.0.0 - Research Prototype</p>
      </footer>
    </div>
  )
}

export default App