import { useState, useEffect } from "react";
import "./App.css";

// Use relative URL when served from same server, or absolute for dev
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? ''  // Same origin when served from backend
  : 'http://localhost:8000';  // Dev server

function TypingAnimation({ text, speed = 50 }) {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, speed);

      return () => clearTimeout(timeout);
    }
  }, [currentIndex, text, speed]);

  return (
    <span>
      {displayedText}
      <span className="typing-cursor">|</span>
    </span>
  );
}

function App() {
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("text");
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(() => {
    // Get theme from localStorage or default to dark
    const savedTheme = localStorage.getItem('theme');
    return savedTheme || 'dark';
  });

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const analyzeText = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/analyze/text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error("Analysis failed");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze text");
    } finally {
      setLoading(false);
    }
  };

  const analyzeUrl = async () => {
    if (!url.trim()) {
      setError("Please enter a URL");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/analyze/url`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });
      if (!response.ok) throw new Error("URL analysis failed");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze URL");
    } finally {
      setLoading(false);
    }
  };

  const analyzeFile = async () => {
    if (!file) {
      setError("Please select a file");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE_URL}/analyze/file`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("File analysis failed");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze file");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const ext = selectedFile.name.split('.').pop().toLowerCase();
      if (!['pdf', 'txt', 'text'].includes(ext)) {
        setError("Please upload a PDF or text file");
        return;
      }
      setFile(selectedFile);
      setError(null);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 0.7) return "#ef4444"; // Red - high misinformation
    if (score >= 0.4) return "#f59e0b"; // Orange - moderate
    return "#10b981"; // Green - low misinformation
  };

  const getLabelColor = (label) => {
    if (label === "False") return "#ef4444";
    if (label === "Misleading") return "#f59e0b";
    if (label === "Unverified") return "#fbbf24";
    return "#10b981";
  };

  return (
    <div className="app-container">
      <button className="theme-toggle" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
      <header className="app-header">
        <h1>Deminsify</h1>
        <p className="tagline">Demistifying medical misinformation</p>
        <p className="subtitle">
          <TypingAnimation text="Detect medical misinformation in text, URLs, or files." speed={50} />
        </p>
      </header>

      <div className="main-content">
        <div className="tabs">
          <button
            className={`tab ${activeTab === "text" ? "active" : ""}`}
            onClick={() => setActiveTab("text")}
          >
            Text
          </button>
          <button
            className={`tab ${activeTab === "url" ? "active" : ""}`}
            onClick={() => setActiveTab("url")}
          >
            URL
          </button>
          <button
            className={`tab ${activeTab === "file" ? "active" : ""}`}
            onClick={() => setActiveTab("file")}
          >
            File Upload
          </button>
        </div>

        <div className="input-section">
          {activeTab === "text" && (
            <>
              <textarea
                rows={10}
                className="text-input"
                placeholder="Paste medical text here to analyze for misinformation..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <button
                className="analyze-button"
                onClick={analyzeText}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze Text"}
              </button>
            </>
          )}

          {activeTab === "url" && (
            <>
              <input
                type="url"
                className="url-input"
                placeholder="Enter URL to analyze (e.g., https://example.com/article)"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
              <button
                className="analyze-button"
                onClick={analyzeUrl}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze URL"}
              </button>
            </>
          )}

          {activeTab === "file" && (
            <>
              <div className="file-upload-area">
                <input
                  type="file"
                  id="file-input"
                  accept=".pdf,.txt,.text"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                />
                <label htmlFor="file-input" className="file-upload-label">
                  {file ? file.name : "Choose PDF or Text File"}
                </label>
                {file && (
                  <button
                    className="remove-file-button"
                    onClick={() => setFile(null)}
                  >
                    ×
                  </button>
                )}
              </div>
              <button
                className="analyze-button"
                onClick={analyzeFile}
                disabled={loading || !file}
              >
                {loading ? "Analyzing..." : "Analyze File"}
              </button>
            </>
          )}
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && (
          <div className="results-section">
            <h2>Analysis Results</h2>
            
            <div className="score-cards">
              <div className="score-card">
                <div className="score-label">Misinformation Risk</div>
                <div
                  className="score-value"
                  style={{ color: getScoreColor(result.misinfo_score) }}
                >
                  {(result.misinfo_score * 100).toFixed(1)}%
                </div>
                <div className="score-bar">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${result.misinfo_score * 100}%`,
                      backgroundColor: getScoreColor(result.misinfo_score),
                    }}
                  />
                </div>
              </div>

              <div className="score-card">
                <div className="score-label">Accuracy Score</div>
                <div
                  className="score-value"
                  style={{ color: getScoreColor(1 - (result.accuracy_score || 0)) }}
                >
                  {((result.accuracy_score || 0) * 100).toFixed(1)}%
                </div>
                <div className="score-bar">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${(result.accuracy_score || 0) * 100}%`,
                      backgroundColor: "#10b981",
                    }}
                  />
                </div>
              </div>
            </div>

            {result.overall_assessment && (
              <div className="assessment">
                <h3>Overall Assessment</h3>
                <p>{result.overall_assessment}</p>
              </div>
            )}

            {result.flagged_sentences && result.flagged_sentences.length > 0 && (
              <div className="flagged-sentences">
                <h3>Flagged Claims</h3>
                {result.flagged_sentences.map((item, index) => (
                  <div key={index} className="flagged-item">
                    <div className="flagged-sentence">{item.sentence}</div>
                    <div className="flagged-meta">
                      <span
                        className="flagged-label"
                        style={{ backgroundColor: getLabelColor(item.label) }}
                      >
                        {item.label}
                      </span>
                      <span className="flagged-confidence">
                        Confidence: {(item.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {result.source_url && (
              <div className="source-info">
                <strong>Source URL:</strong> <a href={result.source_url} target="_blank" rel="noopener noreferrer">{result.source_url}</a>
              </div>
            )}

            {result.filename && (
              <div className="source-info">
                <strong>File:</strong> {result.filename} ({result.file_type?.toUpperCase() || 'Unknown'})
                {result.text_length && (
                  <span> - {result.text_length.toLocaleString()} characters</span>
                )}
              </div>
            )}

            {result.extracted_text_length && (
              <div className="source-info">
                <strong>Extracted Text:</strong> {result.extracted_text_length.toLocaleString()} characters
              </div>
            )}

            <div className="reload-message">
              <p>
                💡 <strong>Tip:</strong> Reload the page to check a different article, piece of text, or file.
              </p>
              <button 
                className="reload-button"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </button>
            </div>
          </div>
        )}

        {!result && !loading && (
          <div className="info-message">
            <p>
              💡 <strong>Tip:</strong> After analyzing content, reload the page to check a different article, piece of text, or file.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
