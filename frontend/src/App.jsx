import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function analyzeSentiment(event) {
    event.preventDefault();

    const text = review.trim();

    if (!text) {
      setError("Please enter a movie review first.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to analyze this review.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the API. Check that FastAPI is running on port 8000."
          : err.message,
      );
    } finally {
      setLoading(false);
    }
  }

  function clearReview() {
    setReview("");
    setResult(null);
    setError("");
  }

  const isPositive = result?.sentiment === "POSITIVE";

  

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">S</span>
          <span>
            Sentiment<span className="brand-light">AI</span>
          </span>
        </a>
        <span className="status">
          <span className="status-dot" />
          DistilBERT API
        </span>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">
            <span>✦</span> NLP · DEEP LEARNING
          </div>
          <h1>
            Understand the feeling
            <br />
            <span>behind every review.</span>
          </h1>
          <p className="hero-description">
            Analyze movie reviews with an AI-powered sentiment classifier. Enter
            your text and discover whether the sentiment is positive or
            negative.
          </p>
        </section>

        <section className="workspace">
          <form className="review-card" onSubmit={analyzeSentiment}>
            <div className="card-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Your movie review</h2>
              </div>
              <span className="text-badge">TEXT ANALYSIS</span>
            </div>

            <label className="input-label" htmlFor="review">
              ENTER REVIEW TEXT
            </label>

            <textarea
              id="review"
              value={review}
              onChange={(event) => {
                setReview(event.target.value.slice(0, 5000));
                setError("");
              }}
              placeholder="For example: The story was brilliant, the actors were amazing, and I loved every minute..."
              maxLength={5000}
              rows={7}
            />

            <div className="input-footer">
              <span>Be as detailed as you like.</span>
              <span>{review.length}/5000</span>
            </div>

            {error && (
              <div className="error-message" role="alert">
                <span>!</span> {error}
              </div>
            )}

            <div className="button-row">
              <button
                className="analyze-button"
                type="submit"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Analyze sentiment <span>→</span>
                  </>
                )}
              </button>

              <button
                className="clear-button"
                type="button"
                onClick={clearReview}
                disabled={loading}
              >
                Clear
              </button>
            </div>
          </form>

          <section className="result-card" aria-live="polite">
            <div className="card-heading">
              <div>
                <span className="step-label">STEP 02</span>
                <h2>Analysis result</h2>
              </div>
              <span className="result-symbol">✳</span>
            </div>

            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-icon">✧</div>
                <h3>Your result will appear here</h3>
                <p>
                  Submit a review to see its predicted sentiment and model
                  confidence.
                </p>
                <div className="empty-divider" />
                <div className="mini-stat">
                  <span>MODEL</span>
                  <strong>DistilBERT</strong>
                </div>
                <div className="mini-stat">
                  <span>TASK</span>
                  <strong>Binary classification</strong>
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <span className="large-spinner" />
                <h3>Reading between the lines...</h3>
                <p>Your review is being analyzed by the model.</p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <div
                  className={`sentiment-icon ${
                    isPositive ? "positive" : "negative"
                  }`}
                >
                  {isPositive ? "↑" : "↓"}
                </div>

                <span className="result-caption">PREDICTED SENTIMENT</span>

                <h3 className={isPositive ? "positive-text" : "negative-text"}>
                  {result.sentiment}
                </h3>

                <div className="confidence-row">
                  <span>Model confidence</span>
                  <strong>{Number(result.confidence).toFixed(2)}%</strong>
                </div>

                <div
                  className="confidence-track"
                  role="progressbar"
                  aria-label="Model confidence"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Number(result.confidence)}
                >
                  <div
                    className={`confidence-fill ${
                      isPositive ? "positive-fill" : "negative-fill"
                    }`}
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, Number(result.confidence) || 0),
                      )}%`,
                    }}
                  />
                </div>

                <div className="review-summary">
                  <span>ANALYZED TEXT</span>
                  <p>{result.text}</p>
                </div>

                <p className="disclaimer">
                  Confidence is the model's estimate, not a guarantee that the
                  prediction is correct.
                </p>
              </div>
            )}
          </section>
        </section>

        <section className="feature-strip">
          <div className="feature">
            <span className="feature-icon">◎</span>
            <div>
              <strong>Transformer-powered</strong>
              <p>DistilBERT NLP model</p>
            </div>
          </div>
          <div className="feature">
            <span className="feature-icon">↗</span>
            <div>
              <strong>Real-time analysis</strong>
              <p>FastAPI REST endpoint</p>
            </div>
          </div>
          <div className="feature">
            <span className="feature-icon">⌘</span>
            <div>
              <strong>Simple and clear</strong>
              <p>Instant prediction results</p>
            </div>
          </div>
        </section>

        <footer>
          Built with React, FastAPI, PyTorch and Hugging Face Transformers.
        </footer>
      </main>
    </div>
  );
}

export default App;
