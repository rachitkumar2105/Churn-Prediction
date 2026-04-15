import { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, ShieldAlert, Cpu, Lightbulb, UserCheck, TrendingUp, BarChart3, UploadCloud, Users, ArrowUpRight } from 'lucide-react'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'
import './index.css'

const COLORS = ['#10b981', '#f59e0b', '#ef4444']; // Low, Medium, High

function App() {
  const [activeTab, setActiveTab] = useState('dashboard') // dashboard, single, bulk
  
  // Dashboard State
  const [analytics, setAnalytics] = useState(null)

  // Single Predict State
  const [formData, setFormData] = useState({
    tenure: 1,
    TotalCharges: 50.0,
    Contract: 'Month-to-month',
    InternetService: 'Fiber optic',
    PaymentMethod: 'Electronic check',
    PaperlessBilling: 'Yes',
    MonthlyCharges: 50.0
  })
  const [modelType, setModelType] = useState('xgboost')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  // Bulk Upload State
  const [file, setFile] = useState(null)
  const [bulkResults, setBulkResults] = useState([])

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchAnalytics();
    }
  }, [activeTab])

  const fetchAnalytics = async () => {
    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await axios.get(`${API_BASE_URL}/analytics`);
      if (res.data.status === 'success') {
        const sortedDist = res.data.data.risk_distribution.sort((a,b) => {
          if(a.name === 'High') return 1;
          if(a.name === 'Low') return -1;
          return 0;
        })
        setAnalytics({...res.data.data, risk_distribution: sortedDist})
      }
    } catch (e) {
      console.error(e)
    }
  }

  const handleChange = (e) => {
    const { name, value } = e.target;
    const numFields = ['tenure', 'TotalCharges', 'MonthlyCharges'];
    setFormData(prev => ({
      ...prev,
      [name]: numFields.includes(name) ? Number(value) : value
    }));
  }

  const handlePredictSingle = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        customer_data: formData,
        model_type: modelType,
        explain: true
      });

      if (response.data.status === 'success') {
        setResult(response.data.data);
      } else {
        setError(response.data.message || 'Error occurred during prediction');
      }
    } catch (err) {
      setError(err.message || 'Server error. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }

  const handleFileUpload = async (e) => {
    const selected = e.target.files[0];
    if (selected) setFile(selected);
  }

  const handleBulkSubmit = async () => {
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("model_type", modelType);

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await axios.post(`${API_BASE_URL}/bulk_predict`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      if (res.data.status === 'success') {
        setBulkResults(res.data.data);
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="crm-layout">
      {/* Sidebar Navigation */}
      <div className="sidebar fade-in">
        <div className="sidebar-logo">
          <Activity size={32} color="#3b82f6" />
          <h1>IntelliChurn</h1>
        </div>
        <div className="sidebar-nav">
          <div className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <BarChart3 size={20} /> Advanced Analytics
          </div>
          <div className={`nav-item ${activeTab === 'single' ? 'active' : ''}`} onClick={() => setActiveTab('single')}>
            <UserCheck size={20} /> Single Customer Profile
          </div>
          <div className={`nav-item ${activeTab === 'bulk' ? 'active' : ''}`} onClick={() => setActiveTab('bulk')}>
            <UploadCloud size={20} /> Batch CSV Scoring
          </div>
        </div>
      </div>

      <div className="main-content">
        <header className="top-bar fade-in">
          <h2>{activeTab === 'dashboard' ? 'Overview Dashboard' : activeTab === 'single' ? 'Customer Analysis' : 'Batch CRM Processing'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>AI Model Engine:</span>
            <div style={{ display: 'flex', gap: '0.5rem', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '8px' }}>
              <button 
                className={`tab ${modelType === 'xgboost' ? 'active' : ''}`}
                onClick={() => setModelType('xgboost')}
                style={{ padding: '0.4rem 1rem', marginBottom: 0, borderRadius: '6px' }}
              >
                XGBoost
              </button>
              <button 
                className={`tab ${modelType === 'nn' ? 'active' : ''}`}
                onClick={() => setModelType('nn')}
                style={{ padding: '0.4rem 1rem', marginBottom: 0, borderRadius: '6px' }}
              >
                Deep Learning
              </button>
            </div>
          </div>
        </header>

        {/* DASHBOARD TAB */}
        {activeTab === 'dashboard' && (
          <div className="fade-in">
            <div className="dashboard-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: '2rem' }}>
              <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
                <span style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Total Predictions</span>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '2rem', fontWeight: '700' }}>{analytics?.total_predictions || 0}</span>
                  <Activity size={32} color="var(--accent-blue)" />
                </div>
              </div>
              <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
                <span style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>High Risk Customers</span>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--danger)' }}>
                    {analytics?.risk_distribution?.find(r => r.name === 'High')?.value || 0}
                  </span>
                  <AlertCircle size={32} color="var(--danger)" />
                </div>
              </div>
              <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
                <span style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Primary Cause of Churn</span>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--warning)' }}>Contract: M2M</span>
                  <Users size={32} color="var(--warning)" />
                </div>
              </div>
            </div>

            <div className="dashboard-grid">
              <div className="card chart-card">
                <h3 className="card-title">Risk Distribution</h3>
                <ResponsiveContainer width="100%" height="80%">
                  <PieChart>
                    <Pie data={analytics?.risk_distribution || []} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={5} dataKey="value">
                      {analytics?.risk_distribution?.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', border: 'none' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="card chart-card">
                <h3 className="card-title">Prediction Trend (Last 7 Days)</h3>
                <ResponsiveContainer width="100%" height="80%">
                  <BarChart data={analytics?.trend_data || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="date" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', border: 'none' }} />
                    <Bar dataKey="predictions" fill="var(--accent-blue)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* SINGLE CUSTOMER TAB */}
        {activeTab === 'single' && (
          <div className="dashboard-grid fade-in">
            <div className="card" style={{ animationDelay: '0.1s' }}>
              <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <UserCheck size={24} /> Customer Profile
              </h2>
              <form onSubmit={handlePredictSingle}>
                <div className="dashboard-grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="form-group">
                    <label>Tenure (Months)</label>
                    <input type="number" name="tenure" className="form-input" value={formData.tenure} onChange={handleChange} min="0" required />
                  </div>
                  <div className="form-group">
                    <label>Monthly Charges ($)</label>
                    <input type="number" name="MonthlyCharges" className="form-input" value={formData.MonthlyCharges} onChange={handleChange} step="0.01" required />
                  </div>
                  <div className="form-group">
                    <label>Total Charges ($)</label>
                    <input type="number" name="TotalCharges" className="form-input" value={formData.TotalCharges} onChange={handleChange} step="0.01" required />
                  </div>
                  <div className="form-group">
                    <label>Contract Type</label>
                    <select name="Contract" className="form-input" value={formData.Contract} onChange={handleChange}>
                      <option value="Month-to-month">Month-to-month</option><option value="One year">One year</option><option value="Two year">Two year</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Internet Service</label>
                    <select name="InternetService" className="form-input" value={formData.InternetService} onChange={handleChange}>
                      <option value="Fiber optic">Fiber optic</option><option value="DSL">DSL</option><option value="No">No</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Payment Method</label>
                    <select name="PaymentMethod" className="form-input" value={formData.PaymentMethod} onChange={handleChange}>
                      <option value="Electronic check">Electronic check</option><option value="Mailed check">Mailed check</option><option value="Bank transfer (automatic)">Bank transfer (automatic)</option><option value="Credit card (automatic)">Credit card (automatic)</option>
                    </select>
                  </div>
                </div>
                <button type="submit" className="btn btn-primary" disabled={loading} style={{ marginTop: '1rem' }}>
                  {loading ? <><div className="loading-spinner"></div> Analyzing...</> : <><Cpu size={20} style={{ marginRight: '0.5rem' }} /> Predict Churn Risk</>}
                </button>
              </form>
            </div>

            <div className="results-area">
              {result ? (
                <>
                  <div className="card fade-in">
                    <h2 className="card-title">Prediction Result</h2>
                    <div className={`result-box ${result.risk_category.toLowerCase()}`}>
                      <div>
                        <p style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Risk Level</p>
                        <div className={`risk-level ${result.risk_category.toLowerCase()}`}>{result.risk_category} Risk</div>
                      </div>
                      <div className="prob-circle" style={{ '--percent': `${result.churn_probability * 100}%` }}>
                        <div className="prob-value">{Math.round(result.churn_probability * 100)}%</div>
                      </div>
                    </div>
                    <div className="recommendation-box" style={{ marginTop: '1.5rem' }}>
                      <h4><Lightbulb size={20} /> AI Recommendation</h4>
                      <p>{result.recommendation}</p>
                    </div>
                  </div>
                  {result.explanation_image_base64 && (
                    <div className="card fade-in">
                      <h2 className="card-title"><TrendingUp size={24} /> Model Explainability (SHAP)</h2>
                      <img src={`data:image/png;base64,${result.explanation_image_base64}`} alt="SHAP Plot" className="shap-image"/>
                    </div>
                  )}
                </>
              ) : (
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                  <ShieldAlert size={64} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                  <h3>Awaiting Data</h3>
                </div>
              )}
            </div>
          </div>
        )}

        {/* BULK UPLOAD TAB */}
        {activeTab === 'bulk' && (
          <div className="fade-in">
            <div className="card file-upload-box" onClick={() => document.getElementById('csv-upload').click()}>
              <UploadCloud size={48} color="var(--accent-blue)" style={{ marginBottom: '1rem' }} />
              <h3>{file ? file.name : "Drag & Drop Bulk CSV"}</h3>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Upload thousands of records for instant CRM scoring.</p>
              <input type="file" id="csv-upload" accept=".csv" style={{ display: 'none' }} onChange={handleFileUpload} />
            </div>

            {file && (
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
                <button className="btn btn-primary" onClick={handleBulkSubmit} disabled={loading} style={{ width: 'auto' }}>
                  {loading ? 'Processing Batch...' : 'Score Batch Database'}
                </button>
              </div>
            )}

            {bulkResults.length > 0 && (
              <div className="data-table-container fade-in">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Customer ID</th>
                      <th>Tenure (Mos)</th>
                      <th>Contract</th>
                      <th>Risk Level</th>
                      <th>Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bulkResults.slice(0, 10).map((row, idx) => (
                      <tr key={idx}>
                        <td>{row.customerID || `User-${idx+1000}`}</td>
                        <td>{row.tenure}</td>
                        <td>{row.Contract}</td>
                        <td><span className={`badge ${row.risk_category.toLowerCase()}`}>{row.risk_category}</span></td>
                        <td>{(row.churn_probability * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  Showing top 10 results. Export full list to CRM to view all {bulkResults.length} records.
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// Dummy icon to avoid errors if not imported correctly above
const AlertCircle = ShieldAlert;

export default App
