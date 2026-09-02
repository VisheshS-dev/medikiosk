import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000/api';

export default function Dashboard() {
  const [queue, setQueue] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [patientData, setPatientData] = useState(null);
  const [status, setStatus] = useState('Waiting for patient...');
  const [historyData, setHistoryData] = useState(null);
  const [showAllHistory, setShowAllHistory] = useState(false);

  // Fetch queue
  const fetchQueue = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/patients`);
      const patients = await response.json();
      setQueue(patients);
    } catch (error) {
      console.error('Failed to fetch queue', error);
    }
  };

  // Auto-select first patient only if none selected
  useEffect(() => {
    if (queue.length > 0 && !selectedPatientId) {
      setSelectedPatientId(queue[0].patient_id);
    }
  }, [queue, selectedPatientId]);

  // Fetch patient data
  const fetchData = async () => {
    if (!selectedPatientId) return;
    try {
      setStatus('Fetching...');
      const response = await fetch(`${API_BASE_URL}/patient-summary?patient_id=${selectedPatientId}`);
      const data = await response.json();
      setPatientData(data);
      setStatus('Synchronized');
    } catch (error) {
      console.error('Failed to fetch data', error);
      setStatus('Error connecting');
    }
  };

  // Delete patient
  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete patient ${id}?`)) return;
    try {
      await fetch(`${API_BASE_URL}/patients/${id}`, { method: 'DELETE' });
      if (selectedPatientId === id) {
        setSelectedPatientId(null);
        setPatientData(null);
      }
      fetchQueue();
    } catch (error) {
      console.error('Failed to delete patient', error);
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchQueue, 5000);
    fetchQueue();
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedPatientId) {
      fetchData();
      fetchHistory();
      setShowAllHistory(false);
    }
  }, [selectedPatientId]);

  // Fetch ABHA-linked history
  const fetchHistory = async () => {
    if (!selectedPatientId) return;
    try {
      const response = await fetch(`${API_BASE_URL}/patient-history?patient_id=${selectedPatientId}`);
      const data = await response.json();
      setHistoryData(data);
      // If still processing, poll again in 3 seconds
      if (data.filter_status === 'processing') {
        setTimeout(fetchHistory, 3000);
      }
    } catch (error) {
      console.error('Failed to fetch history', error);
    }
  };

  const isEmergency = patientData?.is_emergency;

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--color-bg)' }}>

      {/* Sidebar — Patient Queue */}
      <aside style={{
        width: 300, flexShrink: 0, background: 'var(--color-surface)',
        borderRight: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column'
      }}>
        <div style={{ padding: 'var(--space-5)', borderBottom: '1px solid var(--color-border)' }}>
          <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700 }}>🏥 Patient Queue</h3>
          <p className="caption">{queue.length} patient(s)</p>
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {queue.length === 0 ? (
            <div style={{ padding: 'var(--space-6)', textAlign: 'center' }}>
              <p className="caption">No patients in queue</p>
            </div>
          ) : (
            queue.map(p => (
              <div
                key={p.patient_id}
                onClick={() => setSelectedPatientId(p.patient_id)}
                style={{
                  padding: 'var(--space-4) var(--space-5)',
                  borderBottom: '1px solid var(--color-border-light)',
                  cursor: 'pointer',
                  background: selectedPatientId === p.patient_id ? 'var(--color-primary-50)' : 'transparent',
                  borderLeft: selectedPatientId === p.patient_id ? '3px solid var(--color-primary)' : '3px solid transparent',
                  transition: 'all var(--transition-fast)'
                }}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      {p.is_emergency && <span style={{ color: 'var(--color-danger)' }}>🚨</span>}
                      <span style={{ fontWeight: 700, color: 'var(--color-text)', fontSize: '0.95rem' }}>
                        {p.patient_name || p.patient_id}
                      </span>
                    </div>
                    <div className="caption" style={{ color: 'var(--color-primary)', fontWeight: 600, fontSize: '11px' }}>
                      {p.abha_id ? `ABHA: ${p.abha_id}` : p.patient_id}
                    </div>
                    {(p.age || p.gender) && (
                      <div className="caption" style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>
                        {p.age ? `${p.age} yrs` : ''} {p.gender ? `• ${p.gender}` : ''}
                      </div>
                    )}
                  </div>
                  <button 
                    onClick={(e) => handleDelete(e, p.patient_id)} 
                    style={{ background: 'none', border: 'none', cursor: 'pointer', opacity: 0.6 }}
                    title="Delete patient"
                  >
                    🗑️
                  </button>
                </div>
                <span className="caption" style={{ fontSize: '10px', marginTop: '4px', display: 'block' }}>Arrived: {p.created_at}</span>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-6)' }}>

        {!patientData ? (
          <div className="page-container" style={{ minHeight: 'auto' }}>
            <div className="text-center">
              <div style={{ fontSize: '3rem', marginBottom: 'var(--space-4)' }}>👨‍⚕️</div>
              <h2 className="heading-2 mb-2">MediKiosk Clinical Dashboard</h2>
              <p className="subtitle">Select a patient from the queue to view their clinical summary</p>
            </div>
          </div>
        ) : (
          <div className="animate-fade-in">
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h2 className="heading-2">{patientData.patient_name || 'Clinical Summary'}</h2>
                  {patientData.abha_id && (
                    <span className="badge badge-info" style={{ fontSize: '0.8rem', padding: '4px 10px' }}>
                      ABHA: {patientData.abha_id}
                    </span>
                  )}
                </div>
                <p className="caption">
                  {patientData.age ? `Age ${patientData.age}` : ''} {patientData.gender ? `• ${patientData.gender}` : ''} {patientData.phone ? `• Ph: ${patientData.phone}` : ''} • ID: {patientData.patient_id} • {status}
                </p>
              </div>
              <div className="flex gap-3">
                <button className="btn btn-secondary btn-sm" onClick={fetchData}>🔄 Refresh</button>
                <button className="btn btn-primary btn-sm">✓ Start Consultation</button>
              </div>
            </div>

            {/* Emergency Banner */}
            {isEmergency && (
              <div className="alert alert-danger mb-6" style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700 }}>
                🚨 URGENT MEDICAL ALERT — High Severity Symptoms Detected!
              </div>
            )}

            {/* Draft Banner */}
            <div className="alert alert-warning mb-6">
              ⚠️ AI-GENERATED DRAFT — VERIFY BEFORE CLINICAL USE
            </div>

            {/* AI Clinical Decision Support & Differential Diagnoses */}
            {patientData.clinical_impression && (
              <ClinicalImpressionCard impression={patientData.clinical_impression} patientData={patientData} />
            )}

            {/* Cards Grid */}
            <div className="grid-2" style={{ gap: 'var(--space-4)', alignItems: 'start' }}>

              {/* Left */}
              <div className="flex flex-col gap-4">
                <DashCard title="🎯 Chief Complaint" content={patientData.chief_complaint}
                  highlight severity={patientData.severity} />
                <DashCard title="📋 History of Present Illness" content={patientData.hpi} />
                <DashCard title="📂 Past Medical History" content={patientData.past_medical_history} />
                <DashCard title="👨‍👩‍👧‍👦 Family History" content={patientData.family_history} />
              </div>

              {/* Right */}
              <div className="flex flex-col gap-4">
                <DashCard title="🏃 Personal / Lifestyle" content={patientData.personal_history} />
                <DashCard title="⚠️ Allergies" content={patientData.allergies} />
                <DashCard title="🔍 Review of Systems" content={patientData.review_of_systems} />

                {/* Quick Stats */}
                <div className="card" style={{ padding: 'var(--space-5)' }}>
                  <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)' }}>📊 Assessment</h4>
                  <div className="flex flex-col gap-2">
                    <StatRow label="Severity" value={patientData.severity}
                      badge={patientData.severity === 'High' ? 'badge-danger' : patientData.severity === 'Medium' ? 'badge-warning' : 'badge-success'} />
                    <StatRow label="Duration" value={patientData.duration} />
                    <StatRow label="Emergency"
                      value={patientData.is_emergency ? 'Yes' : 'No'}
                      badge={patientData.is_emergency ? 'badge-danger' : 'badge-success'} />
                  </div>
                </div>

                {/* Documents */}
                {patientData.flagged_lab_values && patientData.flagged_lab_values !== '[]' && (
                  <div className="card" style={{ padding: 'var(--space-5)' }}>
                    <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)' }}>📄 Processed Documents</h4>
                    <DocumentsView data={patientData.flagged_lab_values} patientId={patientData.patient_id} />
                  </div>
                )}
              </div>
            </div>

            {/* AYUSH */}
            {patientData.prakriti && patientData.prakriti !== 'Not assessed' && (
              <div className="card mt-4" style={{ padding: 'var(--space-5)', borderLeft: '4px solid var(--color-success)' }}>
                <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)' }}>🌿 AYUSH Assessment</h4>
                <div className="grid-3">
                  <div><span className="caption">Prakriti</span><p className="body-text">{patientData.prakriti}</p></div>
                  <div><span className="caption">Vikriti</span><p className="body-text">{patientData.vikriti}</p></div>
                  <div><span className="caption">Agni</span><p className="body-text">{patientData.agni}</p></div>
                </div>
              </div>
            )}

            {/* ABHA Past Visit History */}
            {historyData && (historyData.relevant_history?.length > 0 || historyData.other_history?.length > 0) && (
              <div className="mt-4">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                  <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700 }}>📜 Past Visit History (ABHA)</h3>
                  {historyData.abha_id && (
                    <span style={{ background: 'var(--color-primary-50)', color: 'var(--color-primary)', padding: '4px 12px', borderRadius: '20px', fontSize: 'var(--font-size-sm)', fontWeight: 600 }}>
                      ABHA: {historyData.abha_id}
                    </span>
                  )}
                </div>

                {historyData.filter_status === 'processing' && (
                  <div className="alert alert-warning mb-4" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                    <div className="spinner" style={{ width: 18, height: 18 }} />
                    🔄 AI is analyzing past medical history for relevance to today's complaint...
                  </div>
                )}

                {/* Relevant History */}
                {historyData.relevant_history?.length > 0 && (
                  <div style={{ marginBottom: 'var(--space-4)' }}>
                    <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)', color: 'var(--color-success)', fontSize: 'var(--font-size-base)' }}>
                      ⚡ Relevant to Today's Visit ({historyData.relevant_history.length})
                    </h4>
                    <div className="flex flex-col gap-3">
                      {historyData.relevant_history.map(visit => (
                        <VisitCard key={visit.id} visit={visit} isRelevant={true} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Other History (collapsible) */}
                {historyData.other_history?.length > 0 && (
                  <div>
                    <button
                      onClick={() => setShowAllHistory(!showAllHistory)}
                      style={{
                        background: 'none', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)',
                        padding: 'var(--space-2) var(--space-4)', cursor: 'pointer', color: 'var(--color-text-muted)',
                        fontSize: 'var(--font-size-sm)', width: '100%', marginBottom: 'var(--space-3)',
                        transition: 'all var(--transition-fast)'
                      }}
                    >
                      {showAllHistory ? '▼ Hide' : '▶ View All'} Other History ({historyData.other_history.length} visits)
                    </button>
                    {showAllHistory && (
                      <div className="flex flex-col gap-3 animate-fade-in">
                        {historyData.other_history.map(visit => (
                          <VisitCard key={visit.id} visit={visit} isRelevant={false} />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

          </div>
        )}
      </main>
    </div>
  );
}

function DashCard({ title, content, highlight, severity }) {
  const isEmpty = !content || content === 'None reported' || content === 'Not recorded';
  return (
    <div className="card" style={{
      padding: 'var(--space-5)',
      borderLeft: highlight ? `4px solid ${severity === 'High' ? 'var(--color-danger)' : 'var(--color-primary)'}` : undefined,
      opacity: isEmpty ? 0.5 : 1
    }}>
      <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)' }}>{title}</h4>
      <p className="body-text" style={{ whiteSpace: 'pre-wrap' }}>
        {isEmpty ? 'Not reported' : content}
      </p>
    </div>
  );
}

function StatRow({ label, value, badge }) {
  return (
    <div className="flex items-center gap-2">
      <span className="caption" style={{ width: 90 }}>{label}:</span>
      {badge ? (
        <span className={`badge ${badge}`}>{value || 'Unknown'}</span>
      ) : (
        <span className="body-text">{value || 'Unknown'}</span>
      )}
    </div>
  );
}

function DocumentsView({ data, patientId }) {
  try {
    const parsed = JSON.parse(data);
    if (!Array.isArray(parsed) || parsed.length === 0) return <p className="caption">No documents</p>;
    return parsed.map((doc, i) => (
      <div key={i} style={{ marginBottom: 'var(--space-4)', paddingBottom: 'var(--space-4)', borderBottom: i < parsed.length - 1 ? '1px solid var(--color-border)' : 'none' }}>
        <h5 style={{ fontWeight: 700, marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)', color: 'var(--color-primary)' }}>
          📄 {doc.document_type || 'Medical Document'}
        </h5>
        
        <div className="flex items-center gap-4 mb-3">
           <span className="caption">👤 Patient: {patientId}</span>
           <span className="caption">📅 Date: {doc.document_date || 'Unknown'}</span>
        </div>
        
        {doc.summary && (
          <div className="mb-3">
            <span className="caption" style={{ fontWeight: 600, display: 'block' }}>Summary:</span>
            <p className="body-text">{doc.summary}</p>
          </div>
        )}
        
        {doc.diagnoses?.length > 0 && <p className="caption mb-1"><strong>Diagnoses:</strong> {doc.diagnoses.join(', ')}</p>}
        {doc.medications?.length > 0 && <p className="caption mb-1"><strong>Medications:</strong> {doc.medications.join(', ')}</p>}
        {doc.flagged_values?.length > 0 && (
          <p className="caption mt-2" style={{ color: 'var(--color-danger)', fontWeight: 600 }}>
            ⚠️ Flagged Findings: {doc.flagged_values.join(', ')}
          </p>
        )}
        
        {doc.file_url && (
           <a href={doc.file_url} target="_blank" rel="noreferrer" className="btn btn-outline btn-sm mt-3" style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)' }}>
             👁️ View Full Original Document
           </a>
        )}
      </div>
    ));
  } catch {
    return <p className="caption">{data}</p>;
  }
}

function VisitCard({ visit, isRelevant }) {
  return (
    <div
      className="card"
      style={{
        padding: 'var(--space-4) var(--space-5)',
        borderLeft: isRelevant ? '4px solid var(--color-success)' : '4px solid var(--color-border)',
        background: isRelevant ? 'var(--color-surface)' : 'var(--color-bg)',
        opacity: isRelevant ? 1 : 0.85,
        boxShadow: isRelevant ? 'var(--shadow-sm)' : 'none'
      }}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span style={{ fontWeight: 700, fontSize: 'var(--font-size-base)', color: 'var(--color-text)' }}>
            📅 {visit.visit_date}
          </span>
          <span className="badge badge-info" style={{ fontSize: 'var(--font-size-xs)' }}>
            {visit.specialty || 'General OPD'}
          </span>
          {isRelevant && (
            <span className="badge badge-success" style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600 }}>
              ⚡ AI Correlated
            </span>
          )}
        </div>
      </div>

      <div style={{ marginBottom: 'var(--space-2)' }}>
        <span className="caption" style={{ fontWeight: 600 }}>Complaint: </span>
        <span className="body-text" style={{ fontWeight: 600 }}>{visit.chief_complaint}</span>
      </div>

      {visit.relevance_reason && (
        <div
          style={{
            background: isRelevant ? 'rgba(16, 185, 129, 0.08)' : 'var(--color-surface)',
            border: isRelevant ? '1px solid rgba(16, 185, 129, 0.25)' : '1px solid var(--color-border-light)',
            borderRadius: 'var(--radius-md)',
            padding: 'var(--space-2) var(--space-3)',
            marginBottom: 'var(--space-3)',
            fontSize: 'var(--font-size-sm)',
            color: isRelevant ? 'var(--color-success-dark, #065f46)' : 'var(--color-text-muted)'
          }}
        >
          <strong>💡 AI Clinical Relevance:</strong> {visit.relevance_reason}
        </div>
      )}

      {visit.summary && (
        <p className="body-text mb-2" style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
          {visit.summary}
        </p>
      )}

      {visit.diagnoses?.length > 0 && (
        <div className="caption mb-1">
          <strong>Diagnoses: </strong>
          {visit.diagnoses.map((diag, idx) => (
            <span key={idx} style={{ display: 'inline-block', background: 'var(--color-primary-50)', color: 'var(--color-primary)', borderRadius: '4px', padding: '1px 6px', margin: '2px', fontSize: '11px', fontWeight: 600 }}>
              {diag}
            </span>
          ))}
        </div>
      )}

      {visit.medications?.length > 0 && (
        <div className="caption mb-1">
          <strong>Medications: </strong>
          {visit.medications.map((med, idx) => (
            <span key={idx} style={{ display: 'inline-block', background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: '4px', padding: '1px 6px', margin: '2px', fontSize: '11px' }}>
              💊 {med}
            </span>
          ))}
        </div>
      )}

      {visit.flagged_values?.length > 0 && (
        <div className="caption mt-2" style={{ color: 'var(--color-danger)', fontWeight: 600 }}>
          ⚠️ Flagged Labs: {visit.flagged_values.join(' • ')}
        </div>
      )}
    </div>
  );
}

function ClinicalImpressionCard({ impression, patientData }) {
  const [isVerified, setIsVerified] = useState(false);
  const [copiedTest, setCopiedTest] = useState(null);

  if (!impression || (!impression.clinical_synthesis && (!impression.probable_diagnoses || impression.probable_diagnoses.length === 0))) {
    return null;
  }

  const { clinical_synthesis, probable_diagnoses = [], suggested_investigations = [], critical_rule_outs = [] } = impression;

  const handleCopyTest = (test) => {
    navigator.clipboard?.writeText(test);
    setCopiedTest(test);
    setTimeout(() => setCopiedTest(null), 2000);
  };

  const getLikelihoodBadge = (likelihood) => {
    const l = (likelihood || 'medium').toLowerCase();
    if (l === 'high') {
      return {
        label: 'High Likelihood',
        bg: 'rgba(239, 68, 68, 0.12)',
        color: '#dc2626',
        border: '1px solid rgba(239, 68, 68, 0.3)'
      };
    }
    if (l === 'medium' || l === 'moderate') {
      return {
        label: 'Moderate Likelihood',
        bg: 'rgba(245, 158, 11, 0.12)',
        color: '#d97706',
        border: '1px solid rgba(245, 158, 11, 0.3)'
      };
    }
    return {
      label: 'Low / Rule-Out',
      bg: 'rgba(59, 130, 246, 0.12)',
      color: '#2563eb',
      border: '1px solid rgba(59, 130, 246, 0.3)'
    };
  };

  return (
    <div
      className="card mb-6"
      style={{
        padding: 'var(--space-5)',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        background: 'linear-gradient(180deg, rgba(239, 246, 255, 0.75) 0%, var(--color-surface) 100%)',
        boxShadow: '0 4px 14px rgba(37, 99, 235, 0.08)',
        borderRadius: 'var(--radius-lg, 12px)'
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-3)', marginBottom: 'var(--space-4)', borderBottom: '1px solid rgba(59, 130, 246, 0.15)', paddingBottom: 'var(--space-3)' }}>
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: '1.4rem' }}>🧠</span>
            <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.01em' }}>
              AI Clinical Decision Support & Differential Insights
            </h3>
            <span className="badge badge-info" style={{ fontSize: '11px', fontWeight: 700, padding: '3px 8px' }}>
              CDSS v2.0
            </span>
          </div>
          <p className="caption" style={{ color: 'var(--color-text-secondary)' }}>
            Cross-modal synthesis of presenting symptoms, scanned lab reports, and verified ABHA health records
          </p>
        </div>

        {/* Doctor Verification Control */}
        <button
          onClick={() => setIsVerified(!isVerified)}
          className={`btn btn-sm ${isVerified ? 'btn-primary' : 'btn-outline'}`}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontWeight: 600, transition: 'all 0.2s ease' }}
        >
          {isVerified ? '✅ Verified by Doctor' : '👨‍⚕️ Click to Verify Impression'}
        </button>
      </div>

      {/* Clinical Synthesis Executive Summary */}
      {clinical_synthesis && (
        <div
          style={{
            background: 'rgba(255, 255, 255, 0.92)',
            borderLeft: '4px solid var(--color-primary)',
            padding: 'var(--space-3) var(--space-4)',
            borderRadius: '0 var(--radius-md) var(--radius-md) 0',
            marginBottom: 'var(--space-4)',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)'
          }}
        >
          <div className="flex items-center gap-1.5 mb-2">
            <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--color-primary)', letterSpacing: '0.05em' }}>
              Clinical Synthesis Overview
            </span>
          </div>
          <ul style={{ margin: 0, paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {(() => {
              let points = [];
              if (Array.isArray(clinical_synthesis)) {
                points = clinical_synthesis.map(p => typeof p === 'string' ? p.replace(/^[•\-\*]\s*/, '').trim() : String(p));
              } else if (typeof clinical_synthesis === 'string') {
                const lines = clinical_synthesis.split(/\n+/).map(l => l.replace(/^[•\-\*]\s*/, '').trim()).filter(Boolean);
                if (lines.length > 1) {
                  points = lines;
                } else {
                  points = clinical_synthesis.split(/(?<=[.!?])\s+/).map(s => s.replace(/^[•\-\*]\s*/, '').trim()).filter(Boolean);
                }
              }
              if (points.length === 0 && clinical_synthesis) {
                points = [String(clinical_synthesis)];
              }
              return points.map((pt, pIdx) => (
                <li key={pIdx} className="body-text" style={{ fontSize: '0.92rem', lineHeight: 1.45, color: 'var(--color-text)' }}>
                  {pt}
                </li>
              ));
            })()}
          </ul>
        </div>
      )}

      {/* Probable Diagnoses / Differentials */}
      {probable_diagnoses.length > 0 && (
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <h4 style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--color-text)', marginBottom: 'var(--space-3)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>🎯</span> Probable Diagnoses & Clinical Evidence ({probable_diagnoses.length})
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-3)' }}>
            {probable_diagnoses.map((diag, idx) => {
              const badge = getLikelihoodBadge(diag.likelihood);
              return (
                <div
                  key={idx}
                  style={{
                    background: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-md)',
                    padding: 'var(--space-3) var(--space-4)',
                    boxShadow: '0 2px 5px rgba(0,0,0,0.03)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between'
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--color-text)' }}>
                        {diag.condition}
                      </span>
                      <span
                        style={{
                          background: badge.bg,
                          color: badge.color,
                          border: badge.border,
                          padding: '2px 8px',
                          borderRadius: '12px',
                          fontSize: '11px',
                          fontWeight: 700,
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {badge.label}
                      </span>
                    </div>
                    {diag.supporting_evidence && (
                      <p className="caption" style={{ color: 'var(--color-text-secondary)', fontSize: '12px', lineHeight: 1.4 }}>
                        <strong>💡 Evidence:</strong> {diag.supporting_evidence}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Two Column Footer: Suggested Investigations & Critical Rule-Outs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-3)', paddingTop: 'var(--space-2)' }}>
        
        {/* Suggested Investigations */}
        {suggested_investigations.length > 0 && (
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.7)',
              border: '1px solid var(--color-border-light)',
              borderRadius: 'var(--radius-md)',
              padding: 'var(--space-3)'
            }}
          >
            <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--color-text)', display: 'block', marginBottom: '8px' }}>
              🧪 Suggested Diagnostic Workup:
            </span>
            <div className="flex flex-wrap gap-2">
              {suggested_investigations.map((test, idx) => (
                <button
                  key={idx}
                  onClick={() => handleCopyTest(test)}
                  title="Click to copy test name"
                  style={{
                    background: copiedTest === test ? 'var(--color-success-50, #d1fae5)' : 'var(--color-surface)',
                    color: copiedTest === test ? 'var(--color-success-dark, #065f46)' : 'var(--color-primary)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '6px',
                    padding: '3px 10px',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <span>{copiedTest === test ? '✓' : '+'}</span> {test}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Critical Rule-Outs */}
        {critical_rule_outs.length > 0 && (
          <div
            style={{
              background: 'rgba(254, 242, 242, 0.75)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              borderRadius: 'var(--radius-md)',
              padding: 'var(--space-3)'
            }}
          >
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#dc2626', display: 'block', marginBottom: '8px' }}>
              ⚠️ Critical Rule-Outs (High-Risk Conditions):
            </span>
            <div className="flex flex-wrap gap-2">
              {critical_rule_outs.map((ruleOut, idx) => (
                <span
                  key={idx}
                  style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    color: '#b91c1c',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    borderRadius: '6px',
                    padding: '3px 9px',
                    fontSize: '12px',
                    fontWeight: 600
                  }}
                >
                  ⛔ {ruleOut}
                </span>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
